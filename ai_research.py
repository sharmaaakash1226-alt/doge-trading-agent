"""
Modular AI & Research Engine for the Crypto Trading Agent.
Organizes external intelligence into clear API Categories:
  1. News & Headlines (CryptoCompare Free Public API, RSS fallback, optional CryptoPanic)
  2. Macro & Sentiment (Alternative.me Fear & Greed, CoinGecko Global Dominance)
  3. AI Reasoning / LLM Synthesis (Gemini, Groq, OpenAI, Local Ollama, or Built-in Quant-NLP)

All network calls use async httpx so multiple categories and assets run simultaneously.
"""

import asyncio
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
import httpx

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "api_config.json")

DEFAULT_CONFIG = {
    "llm_provider": "auto",  # auto | gemini | groq | openai | ollama | heuristic
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    "gemini_model": "gemini-2.5-flash",
    "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
    "groq_model": "llama-3.3-70b-versatile",
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "openai_model": "gpt-4o-mini",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "cryptopanic_api_key": os.environ.get("CRYPTOPANIC_API_KEY", ""),
    "news_enabled": True,
    "macro_enabled": True,
}

# Simple TTL caches so rapid 4s UI polling doesn't spam free APIs
_CACHE: Dict[str, Dict[str, Any]] = {
    "news": {"ts": 0.0, "data": None},
    "macro": {"ts": 0.0, "data": None},
    "ai": {"ts": 0.0, "data": None},
}

NEWS_TTL = 120.0
MACRO_TTL = 180.0
AI_TTL = 90.0

BULLISH_WORDS = {
    "surge", "soar", "rally", "bull", "bullish", "breakout", "etf", "inflow",
    "adoption", "approval", "approve", "partnership", "record", "ath", "gain",
    "up", "buy", "accumulate", "institutional", "upgrade", "launch", "growth"
}
BEARISH_WORDS = {
    "crash", "plunge", "dump", "bear", "bearish", "hack", "exploit", "sec",
    "lawsuit", "ban", "outflow", "selloff", "liquidation", "liquidated", "drop",
    "fall", "fear", "fraud", "bankrupt", "delay", "reject", "warning", "risk"
}


def load_api_config() -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
        except Exception as e:
            print(f"Config load error: {e}")
    return cfg


def save_api_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_api_config()
    for k, v in updates.items():
        if k in cfg:
            cfg[k] = v
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Config save error: {e}")
    # Invalidate AI cache when provider/keys change
    _CACHE["ai"]["ts"] = 0.0
    _CACHE["news"]["ts"] = 0.0
    return cfg


def get_api_categories_status() -> Dict[str, Any]:
    """Returns active status & capabilities for each API Category."""
    cfg = load_api_config()
    active_llm = _resolve_llm_provider(cfg)
    return {
        "categories": [
            {
                "id": "market_data",
                "name": "Market Data & Orderbook",
                "provider": "Binance Public / CCXT",
                "status": "ACTIVE (FREE)",
                "requires_key": False,
            },
            {
                "id": "news_research",
                "name": "Live Crypto News & Headlines",
                "provider": "CryptoPanic" if cfg.get("cryptopanic_api_key") else "CryptoCompare + RSS (Free)",
                "status": "ACTIVE" if cfg.get("news_enabled", True) else "DISABLED",
                "requires_key": False,
            },
            {
                "id": "macro_sentiment",
                "name": "Macro & Crowd Sentiment",
                "provider": "Alternative.me F&G + CoinGecko Global",
                "status": "ACTIVE (FREE)" if cfg.get("macro_enabled", True) else "DISABLED",
                "requires_key": False,
            },
            {
                "id": "ai_reasoning",
                "name": "AI Synthesis & Research Agent",
                "provider": active_llm.upper(),
                "status": "ACTIVE",
                "requires_key": active_llm in ("gemini", "groq", "openai"),
            },
        ],
        "active_llm": active_llm,
        "configured_keys": {
            "gemini": bool(cfg.get("gemini_api_key")),
            "groq": bool(cfg.get("groq_api_key")),
            "openai": bool(cfg.get("openai_api_key")),
            "cryptopanic": bool(cfg.get("cryptopanic_api_key")),
        },
        "llm_provider_setting": cfg.get("llm_provider", "auto"),
    }


def _score_headline_sentiment(text: str) -> str:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    bull = len(words & BULLISH_WORDS)
    bear = len(words & BEARISH_WORDS)
    if bull > bear:
        return "BULLISH"
    if bear > bull:
        return "BEARISH"
    return "NEUTRAL"


# ---------------------------------------------------------
# CATEGORY 1: NEWS & HEADLINES
# ---------------------------------------------------------
async def fetch_crypto_news(
    client: httpx.AsyncClient, cfg: Dict[str, Any], force_refresh: bool = False
) -> Dict[str, Any]:
    now = time.time()
    if not force_refresh and _CACHE["news"]["data"] and (now - _CACHE["news"]["ts"] < NEWS_TTL):
        return _CACHE["news"]["data"]

    if not cfg.get("news_enabled", True):
        return {"source": "Disabled", "articles": [], "overall_news_sentiment": "NEUTRAL"}

    articles: List[Dict[str, str]] = []
    source_used = "Fallback"

    # 1. Optional CryptoPanic API (if key provided)
    cp_key = cfg.get("cryptopanic_api_key", "").strip()
    if cp_key:
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={cp_key}&public=true&kind=news"
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                results = resp.json().get("results", [])[:8]
                for item in results:
                    title = item.get("title", "")
                    votes = item.get("votes", {})
                    pos = votes.get("positive", 0)
                    neg = votes.get("negative", 0)
                    sent = "BULLISH" if pos > neg else ("BEARISH" if neg > pos else _score_headline_sentiment(title))
                    articles.append({
                        "title": title,
                        "source": item.get("domain", "CryptoPanic"),
                        "sentiment": sent,
                        "url": item.get("url", ""),
                    })
                source_used = "CryptoPanic API"
        except Exception as e:
            print(f"CryptoPanic error: {e}")

    # 2. Free CryptoCompare News API (no key required)
    if not articles:
        try:
            resp = await client.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN", timeout=5.0
            )
            if resp.status_code == 200:
                data = resp.json().get("Data", [])[:8]
                for item in data:
                    title = item.get("title", "")
                    articles.append({
                        "title": title,
                        "source": item.get("source_info", {}).get("name", item.get("source", "CryptoCompare")),
                        "sentiment": _score_headline_sentiment(title),
                        "url": item.get("url", ""),
                    })
                source_used = "CryptoCompare Free News"
        except Exception as e:
            print(f"CryptoCompare news error: {e}")

    # 3. Free CoinTelegraph RSS fallback
    if not articles:
        try:
            resp = await client.get("https://cointelegraph.com/rss", timeout=5.0)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                for item in root.findall(".//item")[:8]:
                    title_el = item.find("title")
                    link_el = item.find("link")
                    title = title_el.text.strip() if title_el is not None and title_el.text else ""
                    if title:
                        articles.append({
                            "title": title,
                            "source": "CoinTelegraph RSS",
                            "sentiment": _score_headline_sentiment(title),
                            "url": link_el.text.strip() if link_el is not None and link_el.text else "",
                        })
                source_used = "CoinTelegraph RSS"
        except Exception as e:
            print(f"RSS news error: {e}")

    # Compute aggregate news sentiment
    bull_count = sum(1 for a in articles if a["sentiment"] == "BULLISH")
    bear_count = sum(1 for a in articles if a["sentiment"] == "BEARISH")
    if bull_count > bear_count:
        overall = "BULLISH"
    elif bear_count > bull_count:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    result = {
        "source": source_used,
        "articles": articles,
        "bullish_count": bull_count,
        "bearish_count": bear_count,
        "overall_news_sentiment": overall,
        "updated_at": int(now),
    }
    _CACHE["news"] = {"ts": now, "data": result}
    return result


# ---------------------------------------------------------
# CATEGORY 2: MACRO & CROWD SENTIMENT
# ---------------------------------------------------------
async def fetch_macro_sentiment(
    client: httpx.AsyncClient, cfg: Dict[str, Any], force_refresh: bool = False
) -> Dict[str, Any]:
    now = time.time()
    if not force_refresh and _CACHE["macro"]["data"] and (now - _CACHE["macro"]["ts"] < MACRO_TTL):
        return _CACHE["macro"]["data"]

    if not cfg.get("macro_enabled", True):
        return {"fear_greed_value": 50, "fear_greed_label": "Disabled", "btc_dominance": 0.0}

    fng_val = 50
    fng_label = "Neutral"
    btc_dom = 0.0
    mcap_change_24h = 0.0

    async def _get_fng():
        try:
            r = await client.get("https://api.alternative.me/fng/?limit=1", timeout=4.0)
            if r.status_code == 200:
                entry = r.json()["data"][0]
                return int(entry["value"]), entry["value_classification"]
        except Exception as e:
            print(f"F&G fetch error: {e}")
        return 50, "Neutral"

    async def _get_global():
        try:
            r = await client.get("https://api.coingecko.com/api/v3/global", timeout=4.0)
            if r.status_code == 200:
                gd = r.json().get("data", {})
                dom = float(gd.get("market_cap_percentage", {}).get("btc", 0.0))
                chg = float(gd.get("market_cap_change_percentage_24h_usd", 0.0))
                return round(dom, 2), round(chg, 2)
        except Exception as e:
            print(f"CoinGecko global error: {e}")
        return 0.0, 0.0

    (fng_val, fng_label), (btc_dom, mcap_change_24h) = await asyncio.gather(_get_fng(), _get_global())

    result = {
        "fear_greed_value": fng_val,
        "fear_greed_label": fng_label,
        "btc_dominance_pct": btc_dom,
        "market_cap_change_24h_pct": mcap_change_24h,
        "updated_at": int(now),
    }
    _CACHE["macro"] = {"ts": now, "data": result}
    return result


# ---------------------------------------------------------
# CATEGORY 3: AI REASONING / LLM SYNTHESIS
# ---------------------------------------------------------
def _resolve_llm_provider(cfg: Dict[str, Any]) -> str:
    pref = cfg.get("llm_provider", "auto").lower()
    if pref != "auto":
        return pref
    if cfg.get("gemini_api_key"):
        return "gemini"
    if cfg.get("groq_api_key"):
        return "groq"
    if cfg.get("openai_api_key"):
        return "openai"
    return "heuristic"


def _build_research_prompt(
    signals: List[Dict[str, Any]], news: Dict[str, Any], macro: Dict[str, Any]
) -> str:
    asset_lines = [
        f"- {s['symbol']}: Price ${s['current_price']}, Signal={s['signal']}, RSI={s['rsi']}, "
        f"Entry=${s['entry_price']}, SL=${s['stop_loss']}, TP1=${s['target_1']}"
        for s in signals
    ]
    headline_lines = [f"- [{a['sentiment']}] {a['title']}" for a in news.get("articles", [])[:5]]
    return (
        "You are a concise quantitative crypto research analyst. Analyze the simultaneous market data, "
        "macro sentiment, and breaking headlines below. Output a short, actionable 3-sentence briefing:\n"
        "1) Market Regime & Macro Bias\n"
        "2) Top Setup Opportunity among the scanned assets (if any) or why to stay patient\n"
        "3) Key News Catalyst or Risk Warning.\n\n"
        f"MACRO: Fear & Greed = {macro.get('fear_greed_value')} ({macro.get('fear_greed_label')}), "
        f"BTC Dominance = {macro.get('btc_dominance_pct')}%, 24h MCap Change = {macro.get('market_cap_change_24h_pct')}%\n\n"
        "SIMULTANEOUS TECHNICALS:\n" + "\n".join(asset_lines) + "\n\n"
        "LATEST HEADLINES:\n" + ("\n".join(headline_lines) if headline_lines else "No headlines fetched.")
    )


def _heuristic_synthesis(
    signals: List[Dict[str, Any]], news: Dict[str, Any], macro: Dict[str, Any]
) -> str:
    fng = macro.get("fear_greed_value", 50)
    fng_lbl = macro.get("fear_greed_label", "Neutral")
    news_sent = news.get("overall_news_sentiment", "NEUTRAL")
    active = [s for s in signals if s.get("signal") != "NEUTRAL"]

    if active:
        top = max(active, key=lambda x: x.get("confidence_pct", 0))
        setup_str = (
            f"Strongest technical confluence on {top['symbol']} ({top['signal']}, "
            f"Conf {top['confidence_pct']}%, Entry ${top['entry_price']}, SL ${top['stop_loss']}, TP1 ${top['target_1']})."
        )
    else:
        setup_str = "All scanned pairs are currently in consolidation (NEUTRAL); wait for an EMA/RSI breakout."

    alignment = (
        "News flow aligns with risk-on momentum."
        if news_sent == "BULLISH"
        else ("Headlines lean cautious/bearish; tighten stops and manage leverage." if news_sent == "BEARISH" else "News flow is mixed/neutral.")
    )
    return f"Macro: Fear & Greed at {fng} ({fng_lbl}). {setup_str} {alignment}"


async def generate_ai_briefing(
    client: httpx.AsyncClient,
    cfg: Dict[str, Any],
    signals: List[Dict[str, Any]],
    news: Dict[str, Any],
    macro: Dict[str, Any],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    now = time.time()
    if not force_refresh and _CACHE["ai"]["data"] and (now - _CACHE["ai"]["ts"] < AI_TTL):
        return _CACHE["ai"]["data"]

    provider = _resolve_llm_provider(cfg)
    prompt = _build_research_prompt(signals, news, macro)
    summary = ""
    used_provider = provider

    if provider == "gemini" and cfg.get("gemini_api_key"):
        try:
            model = cfg.get("gemini_model", "gemini-2.5-flash")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={cfg['gemini_api_key']}"
            body = {"contents": [{"parts": [{"text": prompt}]}]}
            r = await client.post(url, json=body, timeout=10.0)
            if r.status_code == 200:
                summary = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Gemini API error: {e}")

    elif provider in ("groq", "openai"):
        key = cfg.get("groq_api_key") if provider == "groq" else cfg.get("openai_api_key")
        url = (
            "https://api.groq.com/openai/v1/chat/completions"
            if provider == "groq"
            else "https://api.openai.com/v1/chat/completions"
        )
        model = cfg.get("groq_model") if provider == "groq" else cfg.get("openai_model")
        if key:
            try:
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 220}
                r = await client.post(url, headers=headers, json=body, timeout=10.0)
                if r.status_code == 200:
                    summary = r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"{provider} API error: {e}")

    elif provider == "ollama":
        try:
            ollama_url = cfg.get("ollama_url", "http://localhost:11434").rstrip("/") + "/api/generate"
            body = {"model": cfg.get("ollama_model", "llama3.2"), "prompt": prompt, "stream": False}
            r = await client.post(ollama_url, json=body, timeout=12.0)
            if r.status_code == 200:
                summary = r.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama API error: {e}")

    if not summary:
        summary = _heuristic_synthesis(signals, news, macro)
        used_provider = "Built-in Quant-NLP (Free)"

    result = {
        "provider": used_provider,
        "briefing": summary,
        "updated_at": int(now),
    }
    _CACHE["ai"] = {"ts": now, "data": result}
    return result
