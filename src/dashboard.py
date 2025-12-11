# src/dashboard.py
from pathlib import Path
from datetime import datetime, timedelta, timezone
import os

import pandas as pd
import joblib
import streamlit as st
import requests
import feedparser
import yfinance as yf
from dateutil.parser import parse as parse_date
import altair as alt

from fulltext import fetch_full_text

from dotenv import load_dotenv
load_dotenv() 

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

LABEL_ORDER = [
    "optimistic",
    "pessimistic",
    "neutral_corporate",
    "product_update",
    "strategic_moves",
    "regulatory",
    "hype_sentiment",
    "macro_noise",
]

NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY"))

NEWSAPI_EVERYTHING_URL = "https://newsapi.org/v2/everything"
LANGUAGE = "en"
PAGE_SIZE = 50  

FINANCE_SOURCES_WHITELIST = {
    # --- Tier 1 Global Finance ---
    "Reuters",
    "Bloomberg",
    "CNBC",
    "Yahoo Finance",
    "Financial Times",
    "The Wall Street Journal",
    "MarketWatch",
    "Barron's",

    # --- Equity Research / Investing News ---
    "Seeking Alpha",
    "The Motley Fool",
    "Investor's Business Daily",
    "Zacks Investment Research",
    "Morningstar",
    "TipRanks",
    "ValueWalk",
    "GuruFocus",
    "Simply Wall St",

    # --- Business-Focused Newspapers ---
    "Forbes",
    "Fortune",
    "Business Insider",
    "The Economist",
    "The New York Times",
    "The Washington Post",
    "USA Today Money",
    "Los Angeles Times Business",
    "Chicago Tribune Business",
    "The Guardian Business",
    "The Telegraph Business",
    "The Independent Business",
    "The Times Business",
    "South China Morning Post Business",

    # --- Technology + Markets Coverage ---
    "TechCrunch",
    "The Verge",
    "Wired",
    "VentureBeat",
    "Ars Technica",
    "Protocol",
    "InformationWeek",
    "SiliconANGLE",
    "CRN",

    # --- ETF / Fund Industry ---
    "ETF.com",
    "ETF Trends",
    "Pensions & Investments",
    "Institutional Investor",
    "FundFire",
    "Citywire",
    "InvestmentNews",

    # --- Crypto / Blockchain (only real finance outlets) ---
    "CoinDesk",
    "The Block",
    "CryptoSlate",
    "Decrypt",

    # --- Global Financial Outlets ---
    "Nikkei Asia",
    "Japan Times Business",
    "India Economic Times",
    "Business Standard",
    "Mint",
    "The Hindu Business Line",
    "The Australian Financial Review",
    "The Globe and Mail",
    "BNN Bloomberg",
    "Toronto Star Business",
    "Financial Post",
    "La Repubblica Business",
    "El País Economía",
    "Handelsblatt",
    "Der Spiegel Business",
    "Le Monde Économie",

    # --- Professional / Corporate Insights ---
    "Harvard Business Review",
    "MIT Technology Review",
    "S&P Global Market Intelligence",
    "Moody’s Analytics",
    "Fitch Ratings",
    "Deloitte Insights",
    "PwC Insights",
    "McKinsey Insights",
}

FINANCE_KEYWORDS = [
    # --- Core Market Terms ---
    "stock", "stocks", "equity", "equities",
    "shares", "share price", "share prices",
    "market cap", "market capitalization",
    "float", "public float",
    "short interest", "short seller", "short squeeze",
    "volume", "trading volume", "turnover",
    "market value", "intraday", "premarket", "after hours",

    # --- Earnings & Financials ---
    "earnings", "earnings report", "earnings call",
    "eps", "earnings per share", "adjusted eps",
    "revenue", "sales", "top line", "bottom line",
    "net profit", "net loss",
    "profit", "profits", "operating profit",
    "net income", "operating income",
    "gross income", "gross profit",
    "gross margin", "operating margin", "profit margin",
    "cash flow", "free cash flow", "fcf",
    "ebitda", "adjusted ebitda",
    "ebit", "operating cash flow",
    "cost of goods sold", "cogs",

    # --- Guidance & Forecasting ---
    "guidance", "raises guidance", "cuts guidance",
    "outlook", "forward outlook",
    "forecast", "profit forecast",
    "sales outlook", "revenue forecast",
    "earnings forecast", "earnings outlook",
    "analyst expectations", "consensus estimates",
    "beats expectations", "misses expectations",
    "surpasses expectations", "falls short",
    "q1", "q2", "q3", "q4", "first quarter", "second quarter",
    "fiscal year", "fy2024", "fy2025", "fy2026",

    # --- Analyst Coverage & Price Targets ---
    "analyst", "analysts", "coverage initiated",
    "price target", "price targets", "revised price target",
    "rating", "buy rating", "sell rating", "hold rating",
    "overweight", "underweight", "equal weight",
    "upgrade", "downgrade",
    "initiated with buy", "initiated with hold", "reiterated buy",
    "street estimates", "street high", "street low",
    "market outperform", "market perform", "underperform",

    # --- Corporate Actions ---
    "ipo", "initial public offering",
    "direct listing", "secondary offering",
    "equity offering", "share issuance", "new shares issued",
    "spinoff", "spin-off", "carve-out", 
    "dividend", "dividends", "dividend yield", "dividend payout",
    "buyback", "share repurchase", "repurchase program",
    "stock split", "reverse split",
    "capital raise", "equity raise", "follow-on offering",

    # --- M&A / Strategic Finance ---
    "merger", "acquisition", "acquires", "to acquire",
    "m&a", "strategic review", "strategic alternatives",
    "takeover", "buyout", "leveraged buyout", "lbo",
    "joint venture", "strategic partnership", "collaboration",
    "stake", "takes stake", "minority stake", "majority stake",
    "investment in", "equity investment", "private placement",
    "deal talks", "deal negotiations", "restructuring deal",

    # --- Valuation / Financial Metrics ---
    "valuation", "undervalued", "overvalued",
    "price-to-earnings", "p/e ratio", "forward p/e",
    "peg ratio", "price-to-sales", "p/s ratio",
    "market multiple", "multiples expansion",
    "enterprise value", "ev/ebitda",
    "discounted cash flow", "dcf model",

    # --- Regulatory & Legal Finance ---
    "sec filing", "sec charges", "10-k", "10-q", "8-k",
    "sec investigation", "regulatory probe",
    "antitrust", "anti-trust", "doj",
    "ftc", "competition watchdog",
    "lawsuit", "class action", "litigation",
    "settlement", "penalty", "fine", "fined",
    "consent decree", "whistleblower",

    # --- Corporate Governance ---
    "shareholder meeting", "proxy vote",
    "proxy battle", "activist investor",
    "board of directors", "board member",
    "ceo", "cfo", "coo", "cto", "chief executive",
    "executive leadership", "management change",
    "executive compensation", "succession plan",

    # --- Debt / Credit / Capital Structure ---
    "debt", "leverage", "high leverage", "deleverage",
    "refinancing", "refinance", "maturity wall",
    "bond issuance", "bond sale", "corporate bonds",
    "high-yield bonds", "junk bonds",
    "credit rating", "downgraded by", "upgraded by",
    "default risk", "credit risk",
    "interest expenses", "interest coverage",
    "capital structure", "balance sheet strength",

    # --- Macro / Market Movement Terms ---
    "index", "indices", "benchmark",
    "s&p 500", "nasdaq", "dow jones",
    "market reaction", "market moves",
    "pre-market", "after-hours",
    "volatility", "market volatility", "vix",
    "yield", "treasury yield", "bond yield",
    "fed", "federal reserve", "interest rates", "rate hike",

    # --- Company Operations with Financial Relevance ---
    "cost-cutting", "cost reductions",
    "restructuring", "reorganization",
    "layoffs", "job cuts", "hiring freeze",
    "efficiency program", "expense reduction",
    "capex", "capital expenditure", "opex",
    "supply chain costs", "logistics costs",
    "unit economics", "operating metrics",

    # --- Investor Terms ---
    "institutional investors", "hedge funds",
    "activist investor", "activist pressure",
    "shareholder", "shareholder value",
    "insider buying", "insider selling",
    "insider transactions", "form 4 filing",
    "large block trade", "institutional ownership",
]

SHOPPING_EXCLUDE_KEYWORDS = [
    "black friday",
    "cyber monday",
    "prime day",
    "gift guide",
    "gift ideas",
    "best gifts",
    "deals",
    "deal",
    "discount",
    "sale",
    "sale price",
    "sale ends",
    "save ",
    "% off",
    "off for the holidays",
    "promo code",
    "coupon",
    "shop now",
    "shopping",
    "shoppers",
    "bought this",
    "bought these",
    "space heater",
    "earrings",
    "watch",
    "headphones",
    "camera",
    "security camera",
    "tv",
    "laptop",
    "how to ",
    "step-by-step",
    "in seconds",
    "without spending a dime",
    "for the holidays",
    "gift",
    "gifts",
    "just $",
]

STOCK_LIST = [

    # --- Mega Tech ---
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","NFLX","TSLA","NVDA","AVGO","AMD",
    "ADBE","INTU","CSCO","IBM","HPQ","DELL",

    # --- Semiconductors ---
    "QCOM","INTC","ASML","ADI","MU","TXN","ARM","TSM","AMAT","LRCX","KLAC","ON",
    "MRVL","WDC","SWKS","MCHP","NXPI","CRUS","LSCC",

    # --- AI / Cloud / Software ---
    "CRM","ORCL","NOW","SNOW","MDB","DDOG","NET","PLTR","ZS","OKTA",
    "TEAM","HUBS","PLAN","WORK","DOCU","TWLO","SMAR","RNG",
    "SPLK","ESTC","CFLT","S","AFRM","APPN","AI","BIGC","FROG","PATH",

    # --- E-commerce / Digital Economy ---
    "MELI","SHOP","SEA","BABA","JD","PDD","EBAY","ETSY","POSH","WISH",
    "LZ","DASH","UBER","LYFT","ABNB","BKNG","EXPE","TRIP",

    # --- Finance / Banks ---
    "JPM","BAC","C","GS","MS","WFC","SCHW","BK","USB","PNC","COF","AXP",
    "BLK","TROW","BEN","NTRS","HBAN","ALLY","KEY","FITB",

    # --- Fintech / Payments ---
    "PYPL","SQ","V","MA","COIN","AXP","GPN","FIS","FISV","HUT","RIOT","MARA",

    # --- Consumer / Retail ---
    "WMT","COST","TGT","NKE","HD","LOW","SBUX","MCD","KO","PEP","PG",
    "UL","KMB","GIS","K","KR","DG","DLTR","BJ","ROST","TJX","GPS","BBY",
    "HAS","MAT","YETI","LULU","CPRI","RL","PVH","DECK","CROX","SKX",

    # --- Healthcare / Pharma / Biotech ---
    "UNH","JNJ","PFE","MRK","LLY","ABBV","BMY","TMO","VRTX","GILD","REGN","BIIB",
    "AMGN","ZSAN","VRTX","ILMN","DHR","MDT","SYK","ISRG","EW",
    "RGEN","HALO","INCY","NBIX","SRPT","ALNY","VIR","MRNA","BNTX","XENE","SAGE","VERV",

    # --- Industrials / Manufacturing / Defense ---
    "CAT","DE","GE","BA","MMM","HON","LMT","NOC","RTX","GD","HEI","TXT",
    "EMR","ETN","PH","TT","ROK","XYL","AME","FLS","AOS",

    # --- Energy / Oil & Gas / Renewables ---
    "XOM","CVX","COP","SLB","HAL","BKR","VLO","MPC","PSX",
    "ENB","TRP","SU","CNQ","DVN","EOG","FANG",
    "NEE","DUK","SO","D","AES","ED","PEG","AEP",

    # --- Metals / Materials / Chemicals ---
    "LIN","NUE","FCX","BHP","RIO","AA","CLF","X","STLD","MOS","CF","APD",
    "DD","EMN","CE","ALB","SQM","LYB","OLN",

    # --- Telecom / Media / Entertainment ---
    "DIS","CMCSA","VZ","T","WBD","PARA","ROKU","NFLX","CHTR","LBRDA",
    "SPOT","SIRI","TTWO","EA","ATVI","UBI","SEGA","ROKU","FUBO",

    # --- Travel / Transportation / Logistics ---
    "DAL","AAL","UAL","LUV","ALK","JBLU",
    "UPS","FDX","XPO","ODFL","ZTO","FWRD","UAL","RCL","CCL","NCLH",

    # --- Automotive / EV / EV Batteries ---
    "F","GM","STLA","TM","HMC","RACE",
    "TSLA","RIVN","LCID","NIO","XPEV","LI","FSR",
    "QS","SLDP","ALB","SQM","PTRA",

    # --- Real Estate / REIT ---
    "PLD","O","AMT","DLR","EQIX","SPG","VICI","AVB","EQR","MAA","WELL",
    "MPW","PEAK","BXP","SLG",

    # --- Food, Agri, Commodity Producers ---
    "ADM","BG","TSN","CAG","KHC","HRL","FMC","DE","SMG",

    # --- Global Leaders (Non-US) ---
    "SONY","TM","RACE","SAP","NVO","ASML","BHP","BP","SHEL","UL",
    "HSBC","HDB","INFY","CS","DB","UBS","NVS","AZN","RHHBY",

    # --- Latin America / Emerging Markets ---
    "VALE","PBR","ITUB","BSBR","SQM","CCU","SID","GGB",
    "TCS","RELIANCE","TATASTEEL","HDFCBANK","INFY","WIT",

    # --- China Tech / Asia Tech ---
    "BIDU","NTES","TCEHY","IQ","HUYA","BILI","KWEB",
    "SMICY","HIMX","VIOT","TCOM","YUMC","ZTO","BEKE",

    # --- Misc Highly Traded Names ---
    "CAR","CVNA","GME","AMC","BBBY","BB","FSLR","SPWR",
    "RUN","ENPH","SEDG","BLNK","CHPT","PLUG",
    "NKLA","WWE","RBLX","AFRM","UPST","SOFI","HOOD","ZI",
    "DOCS","CLOV","OPEN","CVS","CI","HUM"
]

def is_finance_article(title: str, description: str, source: str = "") -> bool:
    """
    Heuristic: keep only stock/finance/business pieces,
    drop pure shopping / consumer how-to / generic lifestyle.
    """
    title = (title or "").lower()
    desc = (description or "").lower()
    text = f"{title} {desc}"

    source_is_finance = source in FINANCE_SOURCES_WHITELIST
    has_finance_kw = any(k in text for k in FINANCE_KEYWORDS)
    looks_shopping = any(bad in text for bad in SHOPPING_EXCLUDE_KEYWORDS)

    if not source_is_finance and not has_finance_kw:
        return False

    if looks_shopping and not source_is_finance:
        return False

    return source_is_finance or has_finance_kw

def fetch_newsapi_everything(query: str, days_back: int):
    """
    Fetch recent articles for a free-text query using NewsAPI /v2/everything.

    Important: free NewsAPI dev accounts are limited to 100 results total.
    We cap at 100 and never request pages past that offset.
    """
    if not NEWSAPI_KEY:
        return []

    now_utc = datetime.now(timezone.utc)
    from_dt = now_utc - timedelta(days=days_back)
    from_date = from_dt.isoformat(timespec="seconds")
    to_date = now_utc.isoformat(timespec="seconds")

    MAX_RESULTS = 100      
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
           
            return all_articles

        if resp.status_code != 200:
            
            return all_articles

        data = resp.json()
        if data.get("status") != "ok":
          
            return all_articles

        articles = data.get("articles", [])
        if not articles:
            break

        for a in articles:
            url = a.get("url", "") or ""

            all_articles.append(
                {
                    "date": a.get("publishedAt", ""),
                    "source": (a.get("source") or {}).get("name", ""),
                    "title": a.get("title", "") or "",
                    "description": a.get("description", "") or "",
                    "url": url,
                    "source_type": "newsapi_everything",
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



def fetch_google_news_rss(query: str):
    """
    Fetch headlines via Google News RSS for a free-text query.
    Uses a browser-like User-Agent so Google is less likely to block us.
    """
    from urllib.parse import quote_plus

    q_param = quote_plus(query)
    url = (
        f"https://news.google.com/rss/search?q={q_param}"
        "&hl=en-US&gl=US&ceid=US:en"
    )

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
        st.warning(f"[Google RSS] Request failed: {e}")
        return items

    if resp.status_code != 200:
        st.warning(
            f"[Google RSS] HTTP {resp.status_code}. "
            f"Body (truncated): {resp.text[:200]}..."
        )
        return items

    feed = feedparser.parse(resp.content)

    if getattr(feed, "bozo", False):
        st.warning(f"[Google RSS] Parse error: {feed.bozo_exception}")

    if not feed.entries:
        return items

    for entry in feed.entries:
        try:
            published = parse_date(entry.published).isoformat()
        except Exception:
            published = datetime.now(timezone.utc).isoformat()

        items.append(
            {
                "date": published,
                "source": "Google News",
                "title": entry.title,
                "description": entry.get("summary", "") or "",
                "url": entry.link,
                "source_type": "google_rss",
            }
        )
    return items



def fetch_live_news(query: str, days_back: int = 14) -> pd.DataFrame:
    """
    High-level: fetch recent news for the given query from multiple sources,
    scrape full article text, deduplicate, and return as DataFrame with 'text'.
    """
    records = []

    newsapi_records = fetch_newsapi_everything(query, days_back)
    records.extend(newsapi_records)

    rss_records = fetch_google_news_rss(query)
    records.extend(rss_records)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if not df.empty:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        df = df[df["date"] >= cutoff]
        if df.empty:
            return df

    df["description"] = df["description"].fillna("")
    df["title"] = df["title"].fillna("")
    df["url"] = df["url"].fillna("")
    df["source"] = df["source"].fillna("")

    df = df[
        df.apply(
            lambda row: is_finance_article(
                row["title"], row["description"], row.get("source", "")
            ),
            axis=1,
        )
    ]

    if df.empty:
        return df

    full_texts = []
    for _, row in df.iterrows():
        txt = ""
        if row["url"]:
            txt = fetch_full_text(row["url"])
        full_texts.append(txt)

    df["full_text"] = full_texts

    df["text"] = (
        df["title"].astype(str)
        + " "
        + df["description"].astype(str)
        + " "
        + df["full_text"].astype(str)
    ).str.strip()

    df = df.sort_values("date", ascending=False)
    df = df.drop_duplicates(subset=["title", "date"])

    return df

def fetch_price_history(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch daily close prices for the given ticker over the last `days_back` days
    using yfinance.
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        data = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        st.warning(f"Could not fetch price data for {ticker}: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    data = data.reset_index()[["Date", "Close"]]
    data.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
    return data


def load_model():
    model_path = MODELS_DIR / "narrative_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def narrative_sentence(label_counts: pd.Series) -> str:
    if label_counts.sum() == 0:
        return "No headlines found for this query / period."

    percentages = (label_counts / label_counts.sum() * 100).round(1)

    parts = []
    label_friendly = {
        "optimistic": "optimistic",
        "pessimistic": "pessimistic",
        "neutral_corporate": "neutral corporate / housekeeping",
        "product_update": "product / feature updates",
        "strategic_moves": "strategic moves (M&A, partnerships, expansion)",
        "regulatory": "regulatory / legal",
        "hype_sentiment": "hype / narrative pumping",
        "macro_noise": "macro / unrelated noise",
    }

    for label in LABEL_ORDER:
        if label in percentages and percentages[label] > 0:
            readable = label_friendly.get(label, label)
            parts.append(f"{percentages[label]}% {readable}")

    if not parts:
        return "No clear narrative — label distribution is empty."

    return "Narrative for this query: " + ", ".join(parts) + "."



def overall_polarity(label_counts: pd.Series) -> str:
    """
    Very simple overall 'positive vs negative' indicator for your 8 labels.
    Treats:
      - optimistic / pessimistic as directional
      - everything else as neutral-ish context
    """
    total = label_counts.sum()
    if total == 0:
        return "No data to judge sentiment."

    positive = label_counts.get("optimistic", 0)
    negative = label_counts.get("pessimistic", 0)

    neutral_like = (
        label_counts.get("neutral_corporate", 0)
        + label_counts.get("product_update", 0)
        + label_counts.get("strategic_moves", 0)
        + label_counts.get("regulatory", 0)
        + label_counts.get("macro_noise", 0)
    )

    if positive > negative and positive > neutral_like:
        return "Overall narrative is **positive / optimistic**."
    if negative > positive and negative > neutral_like:
        return "Overall narrative is **negative / pessimistic**."
    return "Overall narrative is **mixed / neutral**."

def adjust_label_with_rules(text: str, predicted_label: str) -> str:
    t = (text or "").lower()

    regulatory_keywords = [
        "lawsuit", "sues", "sued", "court", "judge", "regulatory",
        "investigation", "probe", "sec", "doj", "ftc",
        "legal case", "settlement", "hearing", "charges", "antitrust"
    ]
    if any(k in t for k in regulatory_keywords):
        return "regulatory"

    strategic_keywords = [
        "partnership", "partners with", "collaboration", "teams up",
        "agreement", "strategic alliance", "joint venture",
        "cooperate", "deal with", "expands partnership",
        "acquires", "acquisition", "merger", "takeover"
    ]
    if any(k in t for k in strategic_keywords):
        return "strategic_moves"

    bullish_phrases = [
        "strong growth stock",
        "strong buy",
        "top growth stock",
        "why this stock is a buy",
        "why this stock could rally",
        "bullish case",
        "growth opportunity"
    ]
    if predicted_label in ("pessimistic", "neutral_corporate"):
        if any(phrase in t for phrase in bullish_phrases):
            return "optimistic"

    return predicted_label




def main():
    st.set_page_config(
        page_title="Stock Narrative Detector",
        page_icon="📈",
        layout="wide",
    )

    st.markdown(
        """
            <style>
           
            div[data-testid="metric-container"] {
                transform: scale(0.7);
                transform-origin: top left;
            }

          
            div[data-testid="metric-container"] {
                margin-right: 0.5rem;
            }
            </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📈 Stock Narrative Detector")
    st.caption(
        "Fetch recent headlines for a stock or company, classify their narrative tone, "
        "and see how the story in the news is evolving."
    )

    model = load_model()
    if model is None:
        st.error(
            "No trained model found. Run `python3 src/train_model.py` "
            "to train and save `narrative_model.joblib` first."
        )
        return

    with st.sidebar:
        st.header("Settings ⚙️")

        stock_choice = st.selectbox(
            "Choose a stock (or type your own):",
            options=STOCK_LIST,
            index=STOCK_LIST.index("MELI") if "MELI" in STOCK_LIST else 0,
            help="Select a ticker from the list or start typing to search.",
        )

        custom_query = st.text_input(
            "Or enter a custom query",
            value="",
            placeholder="Example: MercadoLibre OR MELI, Amazon stock, Shopify, etc.",
            help="If this is filled, it will override the stock ticker above.",
        )

        query = custom_query.strip() if custom_query.strip() else stock_choice

        days_back = st.slider(
            "Look back (days)",
            min_value=1,
            max_value=30,
            value=14,
            help="How many days of news to consider when fetching headlines.",
        )

        run_button = st.button("🚀 Fetch & analyze")

        st.markdown("---")
        st.caption(
            "Sources: NewsAPI, Google News RSS, and optional full-text scraping.\n"
            "Only finance-related headlines are kept."
        )

    if not run_button:
        st.info("Set your query in the sidebar and click **Fetch & analyze** to start.")
        return

    with st.spinner("Fetching news..."):
        df = fetch_live_news(query, days_back=days_back)

    if df.empty:
        st.warning("No finance-related headlines found for this query and time window.")
        return

    with st.spinner("Classifying narratives..."):
        raw_labels = model.predict(df["text"])

    df["label"] = [
        adjust_label_with_rules(txt, lbl)
        for txt, lbl in zip(df["text"], raw_labels)
    ]


    label_counts = df["label"].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)

    st.markdown(
        f"### Results for: `{query}`  "
        f"<span style='font-size: 0.7rem; color: gray;'>"
        f"(last {days_back} days)</span>",
        unsafe_allow_html=True,
    )
    st.write(f"Found **{len(df)}** finance-related headlines.")

    col1, col2, col3 = st.columns(3)

    def small_metric(label: str, value: str):
        st.markdown(
            f"""
            <p style="font-size:0.8rem; margin:0 0 0.1rem; color:#666;">
                {label}
            </p>
            <p style="font-size:1.2rem; margin:0; font-weight:600; color:#333;">
                {value}
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col1:
        small_metric("Total headlines", str(len(df)))

    with col2:
        if label_counts.sum() > 0:
            top_label = label_counts.idxmax()
            small_metric("Dominant narrative label", top_label)
        else:
            small_metric("Dominant narrative label", "N/A")

    with col3:
        min_date = df["date"].min()
        max_date = df["date"].max()
        if pd.notnull(min_date) and pd.notnull(max_date):
            date_range = f"{min_date.date()} → {max_date.date()}"
        else:
            date_range = "N/A"
        small_metric("Date range", date_range)



    st.markdown("---")

    tab_overview, tab_headlines = st.tabs(["📊 Overview", "📰 Headlines"])

    with tab_overview:
        col_left, col_right = st.columns([1, 1.4])

        with col_left:
            st.subheader("Label Distribution")
            st.bar_chart(label_counts)

            st.subheader("Narrative Summary")
            st.write(narrative_sentence(label_counts))

            st.subheader("Overall Positive / Negative Tone")
            st.write(overall_polarity(label_counts))

        with col_right:
            st.subheader(f"{stock_choice} price (last {days_back} days)")
            price_df = fetch_price_history(stock_choice, days_back)

            if price_df.empty:
                st.caption("No price data available for this ticker.")
            else:
                price_df = price_df.copy()
                price_df["date"] = pd.to_datetime(price_df["date"])
                
                ymin = price_df["close"].min()
                ymax = price_df["close"].max()
                padding = (ymax - ymin) * 0.05  
                ymin -= padding
                ymax += padding

                chart = (
                    alt.Chart(price_df)
                    .mark_line()
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y(
                            "close:Q",
                            title="Close price",
                            scale=alt.Scale(domain=[float(ymin), float(ymax)]),
                        ),
                        tooltip=["date:T", "close:Q"],
                    )
                    .properties(height=300)
                )

                st.altair_chart(chart, use_container_width=True)

    with tab_headlines:
        st.subheader("Headlines (most recent first)")
        st.caption("Browse all articles and click the links to open them.")

        table_df = (
            df[["date", "source", "title", "url", "label"]]
            .sort_values("date", ascending=False)
            .reset_index(drop=True)
            .copy()
        )

        table_df["url_raw"] = table_df["url"]

        table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d %H:%M")

        available_labels = sorted(table_df["label"].dropna().unique().tolist())
        selected_labels = st.multiselect(
            "Filter by narrative label",
            options=available_labels,
            default=available_labels,
            help="Select one or more labels to filter the article list.",
        )

        if selected_labels:
            filtered_df = table_df[table_df["label"].isin(selected_labels)]
        else:
            filtered_df = table_df

        st.markdown(f"Showing **{len(filtered_df)}** articles.")

        st.markdown("### Article list")

        if filtered_df.empty:
            st.info("No articles match the current filter.")
        else:
            for idx, row in filtered_df.iterrows():
                st.markdown(
                    f"**{idx + 1}. {row['title']}**  \n"
                    f"*{row['source']} • {row['date']} • `{row['label']}`*  \n"
                    f"[Open article]({row['url_raw']})",
                )
                st.markdown("---")

        st.markdown("### Table view")
        st.caption("Same data in table form (useful for copying / sorting).")

        table_for_view = filtered_df.copy()
        table_for_view["url"] = table_for_view["url_raw"].apply(
            lambda u: f"[open]({u})" if isinstance(u, str) and u.startswith("http") else ""
        )
        table_for_view = table_for_view.drop(columns=["url_raw"])

        st.dataframe(
            table_for_view,
            use_container_width=True,
        )

        with st.expander("Debug: raw label counts"):
            st.write(label_counts)



if __name__ == "__main__":
    main()
