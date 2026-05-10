"""
LLM-powered news analysis for the trading dashboard.

Sends recent news articles for a ticker to Claude and returns structured
sentiment / impact / summary / trade rationale.

Includes:
- Result caching (don't re-analyze same articles)
- Daily call limit (cost safety)
- Graceful failure (returns None if LLM call fails)
"""

import json
import os
import time
from datetime import datetime, date
from typing import Optional

from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "claude-haiku-4-5-20251001"
MAX_CALLS_PER_DAY = 500  # cost safety: ~$1.50/day max
MIN_ARTICLES_TO_ANALYZE = 1

# In-memory cache: { ticker: { "cache_key": str, "result": dict, "timestamp": str } }
_cache: dict[str, dict] = {}

# Daily call counter resets at UTC midnight
_call_counter = {"date": None, "count": 0}


def _check_and_increment_daily_limit() -> bool:
    """Returns True if we can make another call today, False if at limit."""
    today = date.today().isoformat()
    if _call_counter["date"] != today:
        _call_counter["date"] = today
        _call_counter["count"] = 0
    if _call_counter["count"] >= MAX_CALLS_PER_DAY:
        return False
    _call_counter["count"] += 1
    return True


def get_call_stats() -> dict:
    """Stats endpoint for monitoring."""
    return {
        "date": _call_counter["date"],
        "calls_today": _call_counter["count"],
        "daily_limit": MAX_CALLS_PER_DAY,
        "remaining": max(0, MAX_CALLS_PER_DAY - _call_counter["count"]),
        "cached_tickers": list(_cache.keys()),
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a financial news analyst. Analyze the following news articles about {ticker} and return a structured JSON analysis.

Articles:
{articles_text}

Return ONLY valid JSON with this exact schema (no markdown code fences, no commentary):
{{
  "ticker": "{ticker}",
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_confidence": 0.0 to 1.0,
  "impact": "high" | "medium" | "low",
  "summary": "2-3 sentence synthesis of the recent news for this ticker",
  "key_events": ["event 1", "event 2", "event 3"],
  "trade_rationale": "Brief paragraph (2-3 sentences) on what these events might mean for the stock. NEVER give buy/sell recommendations - just describe the implications.",
  "risk_flags": ["any concerns or contradicting signals"]
}}

Important:
- Be objective. Do not invent facts not in the articles.
- If articles contradict, note it in risk_flags.
- This is for an informational dashboard, NOT financial advice.
- Output JSON only, nothing before or after."""


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def _make_cache_key(articles: list[dict]) -> str:
    """Cache key is the latest article ID (or headline if no ID)."""
    if not articles:
        return "empty"
    latest = articles[0]  # articles are sorted newest first
    return latest.get("id") or latest.get("headline", "")[:80]


def analyze_news(ticker: str, articles: list[dict], force_refresh: bool = False) -> Optional[dict]:
    """
    Analyze news articles for a ticker.

    Returns:
        {
            "analysis": { sentiment, summary, key_events, ... },
            "metadata": { model, input_tokens, output_tokens, elapsed, cached, generated_at },
        }
        OR None if analysis failed or daily limit hit.
    """
    if not articles or len(articles) < MIN_ARTICLES_TO_ANALYZE:
        return None

    cache_key = _make_cache_key(articles)

    # Check cache
    if not force_refresh:
        cached = _cache.get(ticker)
        if cached and cached.get("cache_key") == cache_key:
            result = dict(cached["result"])
            result["metadata"] = dict(result["metadata"])
            result["metadata"]["cached"] = True
            return result

    # Daily limit check
    if not _check_and_increment_daily_limit():
        # Return last cached result if we have one, otherwise None
        cached = _cache.get(ticker)
        if cached:
            result = dict(cached["result"])
            result["metadata"] = dict(result["metadata"])
            result["metadata"]["cached"] = True
            result["metadata"]["limit_reached"] = True
            return result
        return None

    # Build prompt
    articles_text = "\n\n".join(
        f"[{i+1}] {a.get('source', 'unknown')} ({a.get('created_at', '')})\n"
        f"{a.get('headline', '')}\n"
        f"{a.get('summary', '')}"
        for i, a in enumerate(articles[:10])  # cap at 10 articles
    )
    prompt = PROMPT_TEMPLATE.format(ticker=ticker, articles_text=articles_text)

    # Call Claude
    try:
        client = Anthropic()  # uses ANTHROPIC_API_KEY from env
        start = time.time()
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - start

        text = response.content[0].text.strip()

        # Strip code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        analysis = json.loads(text)

        result = {
            "analysis": analysis,
            "metadata": {
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "elapsed_seconds": round(elapsed, 2),
                "estimated_cost": round(
                    (response.usage.input_tokens / 1000 * 0.001)
                    + (response.usage.output_tokens / 1000 * 0.005),
                    5,
                ),
                "cached": False,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "n_articles_analyzed": len(articles[:10]),
            },
        }

        # Cache it
        _cache[ticker] = {"cache_key": cache_key, "result": result}

        return result

    except json.JSONDecodeError as e:
        print(f"LLM returned non-JSON for {ticker}: {e}")
        return None
    except Exception as e:
        print(f"LLM call failed for {ticker}: {e}")
        return None


def clear_cache(ticker: Optional[str] = None):
    """Clear cache for one ticker or all."""
    if ticker:
        _cache.pop(ticker, None)
    else:
        _cache.clear()
