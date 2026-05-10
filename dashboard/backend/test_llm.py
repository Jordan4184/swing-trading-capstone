"""
Standalone test of Claude API for news intelligence.

Tests:
1. API key works
2. Claude returns valid JSON in our expected schema
3. Cost is reasonable
"""

import json
import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()  # uses ANTHROPIC_API_KEY from env

# Sample articles to test with — mimics what Alpaca returns
SAMPLE_ARTICLES = [
    {
        "headline": "NVIDIA Q3 earnings beat consensus by 8 cents; raises Q4 guidance citing accelerating AI demand",
        "summary": "Revenue of $35.1B vs $33.2B expected. Data center segment grew 94% YoY. Management says supply remains constrained through H1 2026.",
        "source": "Reuters",
        "created_at": "2026-05-09T14:30:00Z",
    },
    {
        "headline": "NVDA announces strategic partnership with Boeing for autonomous flight systems",
        "summary": "Multi-year agreement to provide AI compute infrastructure for next-gen aircraft. Financial terms undisclosed but estimated at $400M+.",
        "source": "Bloomberg",
        "created_at": "2026-05-09T13:45:00Z",
    },
    {
        "headline": "Why analysts see NVDA target prices above $1,000 by year-end",
        "summary": "Goldman raises target to $1,050. Morgan Stanley to $985. Wells Fargo cautious at $880 hold rating, citing valuation concerns.",
        "source": "Barron's",
        "created_at": "2026-05-09T11:20:00Z",
    },
]

PROMPT_TEMPLATE = """You are a financial news analyst. Analyze the following news articles about {ticker} and return a structured JSON analysis.

Articles:
{articles_text}

Return ONLY valid JSON with this exact schema (no markdown code fences, no commentary):
{{
  "ticker": "{ticker}",
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_confidence": 0.0 to 1.0,
  "impact": "high" | "medium" | "low",
  "summary": "2-3 sentence synthesis of the day's news for this ticker",
  "key_events": ["event 1", "event 2", "event 3"],
  "trade_rationale": "Brief paragraph (2-3 sentences) on what these events might mean for the stock. NEVER give buy/sell recommendations - just describe the implications.",
  "risk_flags": ["any concerns or contradicting signals"]
}}

Important:
- Be objective. Do not invent facts not in the articles.
- If articles contradict, note it in risk_flags.
- This is for an informational dashboard, NOT financial advice.
- Output JSON only, nothing before or after."""


def analyze_news(ticker: str, articles: list[dict]) -> dict:
    """Send articles to Claude and parse structured response."""
    articles_text = "\n\n".join(
        f"[{i+1}] {a['source']} ({a['created_at']})\n{a['headline']}\n{a['summary']}"
        for i, a in enumerate(articles)
    )
    prompt = PROMPT_TEMPLATE.format(ticker=ticker, articles_text=articles_text)

    start = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start

    text = response.content[0].text.strip()
    # Some models wrap JSON in code fences despite instructions; handle gracefully
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"!! Failed to parse JSON: {e}")
        print(f"Raw response:\n{text}")
        return None

    return {
        "analysis": parsed,
        "metadata": {
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "elapsed_seconds": round(elapsed, 2),
        },
    }


if __name__ == "__main__":
    print("Testing Claude analysis on 3 NVDA articles...\n")
    result = analyze_news("NVDA", SAMPLE_ARTICLES)

    if result is None:
        print("FAILED - see error above")
        exit(1)

    print("=== ANALYSIS ===")
    print(json.dumps(result["analysis"], indent=2))
    print("\n=== METADATA ===")
    print(json.dumps(result["metadata"], indent=2))

    # Estimate cost (Haiku 4.5 pricing as of writing: roughly $0.001 per 1K input, $0.005 per 1K output)
    in_tok = result["metadata"]["input_tokens"]
    out_tok = result["metadata"]["output_tokens"]
    est_cost = (in_tok / 1000 * 0.001) + (out_tok / 1000 * 0.005)
    print(f"\nEstimated cost: ${est_cost:.5f}")
    print(f"At {est_cost:.5f}/call, 11 tickers × 24 calls/day = ${est_cost * 11 * 24:.4f}/day worst case")
