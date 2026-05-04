# Crowd Consensus Meter — Design Document

**Status:** Future enhancement, post-capstone
**Priority:** High (key differentiator for portfolio version)
**Estimated build time:** ~4 weeks

## Concept

A dashboard indicator that shows where retail crowd sentiment sits for any
given ticker, used as an information layer (not a trading trigger) to help
inform contrarian-aware decisions.

Single 0-100 score per ticker, displayed prominently in the dashboard.
Higher = more bullish crowd consensus = potential contrarian sell flag.
Lower = more bearish/ignored = potential contrarian buy opportunity.

## Why this is interesting

- Documented academic effect: extreme retail sentiment historically
  correlates with contrarian forward returns (AAII data, 1987-present).
- Modern incarnation: Reddit/StockTwits/Google Trends sentiment.
- Distinguishes "consensus trades" (crowd already there) from "contrarian
  setups" (crowd absent or wrong).
- Useful even if never traded on directly — informs position sizing and
  timing on every other signal in the system.

## Data sources

| Source                                | Library            | Cost                | What it captures                    |
| ------------------------------------- | ------------------ | ------------------- | ----------------------------------- |
| Reddit (WSB, /r/stocks, /r/investing) | PRAW               | Free                | Mention frequency, attention spikes |
| StockTwits                            | direct API         | Free w/ rate limits | Explicit bull/bear sentiment tags   |
| Google Trends                         | pytrends           | Free (rate-limited) | Search interest, leading indicator  |
| CBOE put/call ratio                   | scrape or yfinance | Free                | Institutional-leaning contrarian    |

## Sub-signals to compute (per ticker, daily)

1. **Reddit attention z-score**: 24hr mention count vs trailing 30-day baseline
2. **StockTwits bull ratio**: bull_count / (bull + bear) over rolling 7d
3. **Search interest z-score**: pytrends interest vs longer baseline
4. **Trend direction**: is sentiment accelerating or decelerating?

## Composite score (0-100)

Weighted average of normalized sub-signals. Weight StockTwits by message
volume (low-volume = unreliable). Output composite + alignment metric
(how aligned the sub-signals are — aligned signals are stronger).

Tier labels:

- 0-20: Extreme Bear (contrarian buy zone)
- 21-40: Bearish
- 41-60: Neutral
- 61-80: Bullish
- 81-100: Extreme Bull (contrarian caution zone)

## Dashboard placement

Card per ticker showing:

- Headline: composite score + tier label + horizontal bar (color gradient)
- Subcomponents on hover: each sub-signal with z-score / ratio
- Alignment indicator
- Optional: contrarian flag with historical mean reversion stat
  ("In similar setups historically: -X% over next 20 days, n=Y instances")

## Decision workflow (how to actually use it)

| Model signal | Crowd consensus | Action                                                     |
| ------------ | --------------- | ---------------------------------------------------------- |
| BUY          | Low (<30)       | High conviction — model + crowd disagree, contrarian setup |
| BUY          | High (>80)      | Yellow flag — easy money may be gone, smaller size         |
| SELL         | High (>80)      | Confirmed — model + crowd both bearish                     |
| SELL         | Low (<20)       | Caution — possible mean reversion incoming                 |
| None         | Extreme (any)   | Watch but don't act on sentiment alone                     |

## Architecture

New module: `src/sentiment.py`

- Data fetchers (one function per source)
- Z-score / normalization helpers
- Composite score computation
- Persistence layer (parquet daily snapshots)

Integration: separate code path from ML model. Both feed dashboard.
Sentiment is dashboard layer, not model feature (yet).

## Build sequence

1. Week 1: Reddit + StockTwits fetchers, daily aggregation, z-scoring
2. Week 2: Google Trends, composite score, persistence
3. Week 3: Streamlit dashboard with meter visualization
4. Week 4: Backtest validation — partition past trades by consensus tier,
   verify contrarian effect actually exists in your universe

## Validation step (CRITICAL — don't skip)

Before trusting the meter:

- Take past model BUY signals from backtest period
- Group by crowd consensus tier at signal time
- Compare forward returns across tiers
- Hypothesis: low-consensus BUYs should outperform high-consensus BUYs
- If hypothesis fails: meter is informational only, not predictive

## Risks / honest concerns

- Sentiment data is noisy; signals may not survive cost-aware backtesting
- Pytrends is unofficial and rate-limits aggressively
- StockTwits sentiment can be brigaded (paid promoters)
- Reddit metrics dominated by 1-2 large names (TSLA, NVDA, etc.)
- "Avoiding low-consensus names because they feel scary" is still herding;
  the meter doesn't fix the discipline problem, only displays it

## Future extensions

- LLM-based qualitative sentiment summary (Claude reads top 5 posts on
  ticker, returns structured "what's the bull/bear case being made")
- Options flow integration (unusual put/call activity)
- News-event reactivity scoring (how much did sentiment move on last
  earnings/major news)
