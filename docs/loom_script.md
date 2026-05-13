# Loom Walkthrough Script

**Target length:** 90 seconds (≈ 240 words at 160 wpm — read briskly, don't rush).
**Tone:** Confident, specific, honest. Lead with process, not numbers.
**Audience:** Hiring engineer or ML interviewer who has 90 seconds and may not click through the repo.

---

## Before you record

- **Browser:** dashboard at `http://localhost:3000/`, signed into paper account so the DecisionCard shows live UNH/today's pick. Confirm the Conviction Ledger has rendered (refresh once if needed).
- **Other tab ready:** `http://localhost:3000/evaluation` scrolled to the **Risk-Layer Ablation** section. Don't pre-scroll past it — you'll move down it during the take.
- **Recording settings:** 1080p, "Screen + Cam" with cam in lower-right, mic on. Mute notifications. Close Slack.
- **Read it through once aloud first.** The phrasing below is written for speech, not the page — pauses fall naturally where the commas are.

---

## 0:00 — 0:08 · Hook

> **[On camera, no screen yet]**
>
> "Hi, I'm Jordan. This is a swing-trading capstone built around one question: when FINRA removes the Pattern Day Trader rule next June, can retail-grade ML produce honest alpha? Let me show you the system — and where it fails."

---

## 0:08 — 0:30 · Today's decision

> **[Cut to dashboard, focus on Decision Card]**
>
> "This is today's pick. Random Forest model, 11-ticker universe, top-2 cross-sectional ranker. The model picks UNH — 62.6% confidence."
>
> **[Mouse hovers over the Calibration Ribbon]**
>
> "The ribbon underneath shows that this probability bucket has historically resolved 37% of the time against a 20% baseline."
>
> **[Mouse moves to the Conviction Ledger]**
>
> "The ledger below is the SHAP attribution. 60-day return is dragging proba down, volatility regime is pushing it up. So I know not just *that* the model picked UNH, but *why*. Recommended size is vol-adjusted: $41K, 108 shares, full position because the SPY/VIX regime gate reads risk-on."

---

## 0:30 — 0:55 · Backtest + honesty

> **[Switch to /evaluation, scroll to Risk-Layer Ablation]**
>
> "Six-year out-of-sample backtest. v1 — the original ranker — does 34% annual return with a 54% drawdown. v2 layers vol-targeted sizing, a regime gate, and a correlation filter. Sharpe goes from 0.95 to 1.09."
>
> **[Hover over the bootstrap CI numbers in the stat bar]**
>
> "Every headline metric has a 95% bootstrap CI. v1's annualized-return CI includes zero — the 34% could have been luck. v2's CI is entirely positive. The risk layer didn't just lift the mean, it pulled the lower bound into safer territory."
>
> **[Tap the ablation table]**
>
> "And the ablation table proves vol-targeting carries almost all of v2's improvement — the correlation filter is actually slightly negative on this sample. That's a finding I publish, not hide."

---

## 0:55 — 1:18 · One honest failure

> **[Scroll to Failure-Mode Case Studies]**
>
> "Here's the model getting it wrong. February 2020, three trading days before COVID: model picks JPM and AMZN, regime reads risk-on, full size. SHAP shows the model correctly read low recent volatility — and got blindsided by the largest 30% drawdown in history."
>
> **[Tap the 'what I'd change' line on the 2020 card]**
>
> "The 200-day moving average regime gate is too lagged for that kind of shock. An implied-vol input would have caught it. I documented two other failure modes the same way — each one a different *kind* of being wrong."

---

## 1:18 — 1:30 · Close

> **[Cut to camera, screen recedes]**
>
> "What I want you to take from this: I audited my own README, found a Sharpe-annualization error, corrected it publicly. Tested cross-sectional rank features, reverted when realized Sharpe degraded despite the AUC win. Every claim on the dashboard ships with a confidence interval or a sample size. The metrics matter — but the process is the point."
>
> **[End frame: GitHub URL + email in lower third]**

---

## After recording

- Trim to within ±5s of 90s. If you went long, the section to cut is "0:30–0:55" — drop the CI sentence and let the ablation table speak for itself.
- Loom embed lives at the top of the README, under the project title and above the Results Summary table.
- Update `README.md` to add the embed once you have the URL. CLAUDE.md doesn't need a change.

---

## Reference numbers for fact-checking before record

| Claim | Source |
|---|---|
| Sharpe 0.95 → 1.09 | `results/backtest_v2_riskmanaged.json` |
| MaxDD -54.1% → -34.3% | same |
| CAGR 34.1% → 24.7% | same |
| v1 CAGR CI includes zero | `backtest_summary.json::strategy_ci.annualized_return` (low ≈ -0.012) |
| v2 CAGR CI entirely positive | `backtest_v2_riskmanaged.json::v2_strategy_ci.annualized_return` (low ≈ +0.049) |
| Vol-targeting carries most of v2 lift | `ablation_v2.json` (Δ Sharpe +0.13, Δ MaxDD +19.7pp) |
| UNH today: 62.6% / 37% / $41K / 108 sh | `/api/risk/today` (numbers will refresh as date advances) |
| Calibration top bucket: 37% at n=1,133 | `/api/calibration/buckets` |
| 2020-02-18 failure: JPM + AMZN, -6.32% | `results/failure_modes.json` |

---

## Optional follow-on (if 90s is too tight)

A 2-3 minute "deep dive" version can re-cut from the same shot, splitting at the natural section breaks. The narration order is already structured for it: hook → decision → backtest → failure → close.
