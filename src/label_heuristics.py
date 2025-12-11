# src/label_heuristics.py
from pathlib import Path
import argparse
import pandas as pd
import hashlib

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

LABELS = [
    "optimistic",
    "pessimistic",
    "neutral_corporate",
    "product_update",
    "strategic_moves",
    "regulatory",
    "hype_sentiment",
    "macro_noise",
]


def heuristic_label(text: str) -> str:
    if not isinstance(text, str):
        text = ""

    t = text.lower()

    optimistic_kw = [
        "soars","jumps","surges","beats expectations","beats estimates","strong growth",
        "rallies","rises","bullish","upgrade","beats forecasts","tops expectations",
        "tops estimates","above expectations","better than expected","smashes estimates",
        "crushes estimates","strong earnings","record revenue","record profits",
        "record earnings","raises guidance","hikes guidance","lifts guidance",
        "guidance raised","strong outlook","positive outlook","upbeat outlook",
        "improving margins","margin expansion","profit jumps","profit surge",
        "revenue jumps","double-digit growth","accelerating growth","strong demand",
        "robust demand","outperforms market","outperforms peers","price target raised",
        "rating upgraded","initiated with buy","buy rating","overweight rating",
        "street high target","beats the street",
        "exceeds expectations", "exceeds forecasts", "top-line beat", "bottom-line beat",
        "strong quarter", "revenue beat", "profit beat", "earnings surprise",
        "positive surprise", "shares climb", "shares gain", "shares advance",
        "upward momentum", "bullish momentum", "breakout", "multi-year high",
        "all-time high", "ath", "strengthening demand", "improving trends",
        "record-breaking", "new highs", "optimistic forecast", "guidance boosted",
        "market share gains", "expanding market share", "growth accelerates",
        "expansion continuing", "operating leverage improving",
        "beats street expectations", "exceeding targets", "forecast raised",
        "profits improve", "expanding margins", "stronger-than-expected",
        "raised outlook", "demand picking up", "momentum builds", "turnaround",
        "positive catalyst", "favorable outlook", "investor optimism",
        "sequential improvement", "recovery continues", "revival", "resilient demand",
        "resilient performance", "revenue strength", "profitability improves",
        "robust performance", "strong performance", "positive trend",
        "above-street estimates", "analyst upgrade", "initiated outperform",
        "price target boosted", "EPS beat", "revenue acceleration", "growth tailwinds",
    ]


    pessimistic_kw = [

        "plunges","falls","slumps","misses estimates","misses expectations","downgrade",
        "bearish","drops","tumbles","misses forecasts","below expectations",
        "worse than expected","disappoints investors","disappointing results",
        "weak earnings","weak results","profit warning","issues profit warning",
        "cuts guidance","slashes guidance","guidance cut","guidance lowered",
        "lowered outlook","weak outlook","downbeat outlook","slowing growth",
        "growth slows","soft demand","weak demand","margin pressure",
        "margin compression","rising costs","cost pressures","regulatory probe",
        "regulatory investigation","antitrust probe","faces scrutiny",
        "accounting scandal","fraud allegations","sec investigation",
        "under investigation","lawsuit","class action","shares sink","shares slide",
        "shares skid","stock skid","stock selloff","sell-off","downgraded to sell",
        "downgraded to neutral","cut to hold","profit slump","warns of slowdown",
        "missed revenue",
        "profit drops","earnings miss","revenue miss","eps miss","warns investors",
        "warns of challenges","weak quarter","misses targets","guidance reduced",
        "reduces outlook","demand weakens","demand deteriorates","market share loss",
        "loses market share","sluggish demand","flat growth","profit falls",
        "revenue declines","shrinking margins","cost overruns","legal troubles",
        "faces lawsuit","regulatory action","settlement talks","whistleblower claims",
        "compliance issues","penalty risk","liquidity concerns","going concern",
        "downgraded outlook","credit downgrade","cut price target",
        "negative catalyst","channel checks weak","inventory buildup",
        "inventory glut","oversupply","layoffs","job cuts","restructuring charges",
        "impairment", "write-down","write off","delays production","supply chain issues",
        "recall", "safety concerns","data breach","cyberattack","hack",
        "governance issues","cash burn","loss widens","expands losses",
        "profitability deteriorates","operating loss increases",
    ]

    macro_kw = [
        "latam economy","emerging markets","em stocks","fed","federal reserve",
        "interest rates","inflation","macro","global markets","risk-off",
        "central bank","rate hike","rate hikes","rate cut","rate cuts",
        "monetary policy","tightening cycle","loosening cycle","recession fears",
        "recession risk","economic slowdown","economic downturn","gdp growth",
        "jobless claims","unemployment rate","economic data","macro headwinds",
        "macro backdrop","currency volatility","forex volatility","strong dollar",
        "weak dollar","bond yields","treasury yields","yield curve","risk-on",
        "market volatility","market turmoil","broad market selloff",
        "indexes tumble","indices tumble","market-wide",
        "dow futures","nasdaq futures","s&p futures","global recession",
        "economic instability","market correction","market crash",
        "banking crisis","credit tightening","macro uncertainty",
        "oil prices surge","oil prices fall","commodity shock",
        "geopolitical tensions","geopolitical risk","war uncertainty",
        "sanctions","trade war","tariffs","supply shock","export restrictions",
        "import restrictions","political turmoil","election uncertainty",
        "fiscal policy","bond market turmoil","inflation expectations",
        "stagflation","deflation concerns","government shutdown risk",
        "manufacturing slowdown","services slowdown","pmi data",
    ]

    hype_kw = [
        "could 10x","next amazon","next amzn","must buy","unstoppable",
        "explosive upside","multi-bagger","multibagger","to the moon","moonshot",
        "skyrocket","set to explode","about to explode","meteoric rise",
        "massive upside","insane upside","life-changing gains",
        "once-in-a-generation","once in a generation","millionaire maker",
        "get rich","can't miss","cant miss","no-brainer buy","no brainer buy",
        "hidden gem","wall street darling","must-own","must own","home run stock",
        "buy the dip","back up the truck","screaming buy",
        "surefire winner","massive breakout","guaranteed gains","slam dunk stock",
        "breakout imminent","explosive potential","tenbagger potential",
        "parabolic move","big money flowing in","crazy upside",
        "insane rally incoming","absolute rocket","monster gains",
        "huge profit potential","get in early","massive momentum",
    ]

    regulatory_kw = [
        "antitrust","anti-trust","doj","department of justice","ftc","sec",
        "regulator","regulatory","regulatory action","regulatory review",
        "regulatory scrutiny","regulatory probe","regulatory investigation",
        "competition authority","competition regulator","cartel","fine","fined",
        "penalty","settlement","consent decree","lawsuit","class action","sued",
        "investigation launched","charges filed",
        "lawsuit", "sues", "sued", "court", "appeal", "judge",
        "regulator", "regulatory", "compliance", "probe", "investigation",
        "settlement", "fine", "fcc", "doj", "ftc", "attorney general",
        "criminal charges","civil charges","regulatory concerns",
        "government inquiry","formal inquiry","regulatory intervention",
        "antitrust lawsuit","blocked merger","deal scrutiny","court ruling",
        "compliance violation","breach of regulations","oversight committee",
    ]

    strategic_kw = [
        "acquisition","acquires","to acquire","merger","merges with","takeover",
        "buyout","joint venture","joint-venture","strategic partnership",
        "partnership with","teams up with","strategic alliance","stake in",
        "takes stake","investment in","invests in","spin-off","spinoff",
        "divestiture","asset sale","strategic review","strategic alternatives",
        "ipo planned","spac deal","raises capital","fundraising round",
        "equity offering","share issuance","capital raise","new subsidiary",
        "enters market","expands into","international expansion",
        "business restructuring","reorganization","new division launched",
    ]

    product_kw = [
        "launches","launch of","rolls out","rollout","introduces","unveils","debuts",
        "new feature","new product","new service","beta version","app update",
        "product update","service update","adds support for","integration with",
        "expands service","expands offering","feature rollout","new tool",
        "pilot program","test program",
        "product refresh","new design","new interface","AI-powered feature",
        "cloud offering","launch event","software update","hardware update",
        "test version","prototype revealed","technology upgrade",
        "improved performance","next-gen version",
    ]

    neutral_corporate_kw = [
        "opens new office","opens new warehouse","opens new fulfillment center",
        "opens new data center","hires","hiring","adds jobs","to hire","staff cuts",
        "job cuts","layoffs","appoints","appointed as","new ceo","new cfo",
        "new coo","leadership change","management change","board member",
        "board of directors","corporate governance","sustainability report",
        "esg report","annual meeting","shareholder meeting","proxy statement",
        "corporate update","company update","operational update","business update",
        "logistics network","supply chain","distribution center","opening facility",
        "closing facility",
        "expands workforce","reduces workforce","facility upgrade",
        "quarterly update","operational metrics","KPI update",
        "hires executive","appoints chairman","restructures team",
        "organizational changes","opens hub","expansion plans",
        "talent acquisition","employment report","corporate filing",
    ]


    if any(k in t for k in optimistic_kw):
        return "optimistic"
    if any(k in t for k in pessimistic_kw):
        return "pessimistic"
    if any(k in t for k in macro_kw):
        return "macro_noise"
    if any(k in t for k in hype_kw):
        return "hype_sentiment"
    if any(k in t for k in regulatory_kw):
        return "regulatory"
    if any(k in t for k in strategic_kw):
        return "strategic_moves"
    if any(k in t for k in product_kw):
        return "product_update"
    if any(k in t for k in neutral_corporate_kw):
        return "neutral_corporate"

    h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
    if h % 2 == 0:
        return "macro_noise"
    else:
        return "neutral_corporate"


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label raw_news_<TICKER>.csv using simple heuristics."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="MELI",
        help="Ticker symbol used in raw_news_<TICKER>.csv (default: MELI).",
    )
    args = parser.parse_args()
    ticker = args.ticker

    raw_path = DATA_DIR / f"raw_news_{ticker}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run collect_news.py first.")

    df = pd.read_csv(raw_path)

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    if "full_text" in df.columns:
        df["full_text"] = df["full_text"].fillna("")
        df["text"] = (
            df["title"] + " " + df["description"] + " " + df["full_text"]
        ).str.strip()
    else:
        df["text"] = (df["title"] + " " + df["description"]).str.strip()

    df["auto_label"] = df["text"].apply(heuristic_label)

    df["label"] = df["auto_label"]

    auto_out = DATA_DIR / f"news_with_auto_labels_{ticker}.csv"
    df.to_csv(auto_out, index=False)
    print(f"Saved {len(df)} rows with auto_label to {auto_out}")

    labeled_out = DATA_DIR / "labeled_news.csv"
    df.to_csv(labeled_out, index=False)
    print(f"Also saved auto-labeled data as {labeled_out} (for training)")


if __name__ == "__main__":
    main()
