# src/fulltext.py
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

def fetch_full_text(url: str, timeout: int = 8) -> str:
    """
    Best-effort fetch of full article text.

    - Returns empty string if anything fails
    - Tries to extract meaningful <p> content
    """
    if not url or not isinstance(url, str):
        return ""

    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException:
        return ""

    ctype = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in ctype:
        return ""

    html = resp.text
    soup = BeautifulSoup(html, "lxml") 

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if p)

    text = re.sub(r"\s+", " ", text).strip()

    return text
