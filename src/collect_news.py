# src/collect_news.py
import os
import csv
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_date
from urllib.parse import quote_plus, urljoin
from pathlib import Path
import argparse

from bs4 import BeautifulSoup
from fulltext import fetch_full_text  

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "53fd7adb63cb4a94b646d285a91be525")

NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"

LANGUAGE = "en"
PAGE_SIZE = 100  


def build_google_news_url(query: str) -> str:
    """
    Build a Google News RSS URL for a free-text query.
    """
    q_param = quote_plus(query)
    return (
        f"https://news.google.com/rss/search?q={q_param}"
        "&hl=en-US&gl=US&ceid=US:en"
    )

def fetch_newsapi_everything(query: str, from_date: str, to_date: str, ticker: str):
    """
    Fetch articles from NewsAPI /v2/everything within the free-plan limits:
    - from_date and to_date within last month
    - to_date at least 24h in the past
    - FREE DEV ACCOUNTS: max 100 results total
    """
    if not NEWSAPI_KEY or NEWSAPI_KEY.startswith("YOUR_"):
        print("No NewsAPI key set, skipping NewsAPI.")
        return []

    MAX_RESULTS = 90        
    PAGE_SIZE = 50       
    all_articles = []
    page = 1

    while True:
        offset = (page - 1) * PAGE_SIZE
        if offset >= MAX_RESULTS:
            break

        params = {
            "q": query,
            "language": LANGUAGE,
            "from": from_date,
            "to": to_date,
            "sortBy": "publishedAt",
            "pageSize": PAGE_SIZE,
            "page": page,
            "apiKey": NEWSAPI_KEY,
        }

        try:
            resp = requests.get(NEWSAPI_EVERYTHING_URL, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"[NewsAPI] Request failed: {e}. Skipping NewsAPI.")
            return all_articles

        if resp.status_code != 200:
            if resp.status_code == 429:
                return all_articles
            try:
                data = resp.json()
                msg = data.get("message", "")
            except Exception:
                msg = resp.text[:200]
            return all_articles


        data = resp.json()
        if data.get("status") != "ok":
            return all_articles

        articles = data.get("articles", [])
        if not articles:
            break

        for a in articles:
            url = a.get("url", "") or ""
            full_text = fetch_full_text(url)

            all_articles.append(
                {
                    "date": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "title": a.get("title", "") or "",
                    "description": a.get("description", "") or "",
                    "url": url,
                    "ticker": ticker,
                    "source_type": "newsapi_everything",
                    "full_text": full_text,
                }
            )

            if len(all_articles) >= MAX_RESULTS:
                return all_articles

        total_results = min(data.get("totalResults", 0), MAX_RESULTS)
        if total_results == 0:
            break

        max_page = (total_results + PAGE_SIZE - 1) // PAGE_SIZE  
        if page >= max_page:
            break

        page += 1

    return all_articles



def fetch_google_news_rss(query: str, ticker: str):
    """
    Fetch articles via Google News RSS for a free-text query.
    """
    url = build_google_news_url(query)
    print(f"[Google RSS] Fetching: {url}")
    items = []

    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            },
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[Google RSS] Request failed: {e}")
        return items

    if resp.status_code != 200:
        print(f"[Google RSS] HTTP status: {resp.status_code}")
        print("[Google RSS] Response (first 300 chars):")
        print(resp.text[:300])
        return items

    feed = feedparser.parse(resp.content)

    if getattr(feed, "bozo", False):
        print("[Google RSS] Parse error:", feed.bozo_exception)

    if not feed.entries:
        print("[Google RSS] No entries returned.")
        return items

    print(f"[Google RSS] Got {len(feed.entries)} entries.")

    for entry in feed.entries:
        try:
            published = parse_date(entry.published).isoformat()
        except Exception:
            published = datetime.now(timezone.utc).isoformat()

        url = entry.link
        full_text = fetch_full_text(url)

        items.append(
            {
                "date": published,
                "source": "Google News",
                "title": entry.title,
                "description": entry.get("summary", "") or "",
                "url": url,
                "ticker": ticker,
                "source_type": "google_rss",
                "full_text": full_text,
            }
        )
    return items


def fetch_yahoo_finance_news(ticker: str, max_items: int = 40):
    """
    Fetch latest news for a ticker from Yahoo Finance's quote news page:
    https://finance.yahoo.com/quote/{ticker}/news?p={ticker}

    If /news 404s or fails, falls back to the main quote page.
    This is HTML scraping (unofficial), but free and usually very relevant.
    """
    base_url = "https://finance.yahoo.com"

    url_news = f"{base_url}/quote/{ticker}/news?p={ticker}"
    url_quote = f"{base_url}/quote/{ticker}?p={ticker}"

    def _fetch_html(url: str):
        print(f"[Yahoo] Fetching: {url}")
        try:
            resp = requests.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                },
                timeout=10,
            )
        except requests.RequestException as e:
            print(f"[Yahoo] Request failed: {e}")
            return None

        if resp.status_code != 200:
            print(f"[Yahoo] HTTP status: {resp.status_code}")
            print("[Yahoo] Response (first 300 chars):")
            print(resp.text[:300])
            return None

        lower_html = resp.text.lower()
        if ("to continue to yahoo" in lower_html) or ("enable javascript" in lower_html):
            print("[Yahoo] Got consent/JS page, cannot reliably scrape.")
            return None

        return resp.text

    html = _fetch_html(url_news)
    if html is None:
        print("[Yahoo] News tab failed, trying main quote page...")
        html = _fetch_html(url_quote)

    if html is None:
        print("[Yahoo] Could not fetch usable HTML for Yahoo Finance.")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    seen_urls = set()
    items = []

    selectors = [
        'section[id^="quoteNewsStream"] h3 a[href*="/news/"]',
        'section[data-test="qsp-news"] h3 a[href*="/news/"]',
        'section[data-test="qsp-major-news"] h3 a[href*="/news/"]',
        'h3 a[href*="/news/"]',
        'a[href*="/news/"]', 
    ]

    links = []
    used_selector = None
    for sel in selectors:
        links = soup.select(sel)
        if links:
            used_selector = sel
            break

    print(f"[Yahoo] Found {len(links)} candidate links (selector={used_selector!r})")

    for a in links:
        title = a.get_text(" ", strip=True)
        if not title:
            continue

        href = a.get("href", "")
        if not href:
            continue

        full_url = href if href.startswith("http") else urljoin(base_url, href)

        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        published = datetime.now(timezone.utc).isoformat()
        full_text = fetch_full_text(full_url)

        items.append(
            {
                "date": published,
                "source": "Yahoo Finance",
                "title": title,
                "description": "",
                "url": full_url,
                "ticker": ticker,
                "source_type": "yahoo_finance",
                "full_text": full_text,
            }
        )

        if len(items) >= max_items:
            break

    print(f"[Yahoo] Collected {len(items)} articles.")
    return items



def save_to_csv(records, path: Path):
    if not records:
        print("No records to save.")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "source",
                "title",
                "description",
                "url",
                "ticker",
                "source_type",
                "full_text",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"Saved {len(records)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect news for training the Stock Narrative Detector."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="MercadoLibre OR MELI",
        help="Free-text search query (e.g., 'MercadoLibre OR MELI').",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="MELI",
        help="Ticker symbol to store in the CSV (e.g., MELI).",
    )
    args = parser.parse_args()

    query = args.query
    ticker = args.ticker

    today = datetime.now(timezone.utc).date()
    to_date = today - timedelta(days=2)      
    from_date = to_date - timedelta(days=28) 

    print(f"Fetching NewsAPI articles for '{query}' from {from_date} to {to_date}...")
    newsapi_records = fetch_newsapi_everything(
        query, from_date.isoformat(), to_date.isoformat(), ticker
    )
    print(f"NewsAPI returned {len(newsapi_records)} articles.")

    print(f"Fetching Google News RSS for '{query}'...")
    rss_records = fetch_google_news_rss(query, ticker)
    print(f"Google News RSS returned {len(rss_records)} articles.")

    print(f"Fetching Yahoo Finance news for ticker '{ticker}'...")
    yahoo_records = fetch_yahoo_finance_news(ticker)
    print(f"Yahoo Finance returned {len(yahoo_records)} articles.")

    all_records = newsapi_records + rss_records + yahoo_records

    seen = set()
    unique_records = []
    for r in all_records:
        key = (r["title"], r["date"])
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    out_path = DATA_DIR / f"raw_news_{ticker}.csv"
    save_to_csv(unique_records, out_path)


if __name__ == "__main__":
    main()
