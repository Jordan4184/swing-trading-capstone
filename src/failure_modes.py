"""
Failure-mode case studies for the capstone defense.

Hand-picks 3 of v2's worst basket returns, each from a distinct kind of
regime shock, and assembles a structured artifact: picks, y_proba per
pick, realized 5-day return per pick, top SHAP contributors that drove
each pick, plus a one-line "what I'd change" written from market context.

Cases chosen so an interviewer can ask "tell me about a time the model
was wrong" and get three different *kinds* of being wrong, not three
flavors of the same kind:

    1. 2020-02-18 — pre-COVID risk-on lock-in (regime gate too lagged)
    2. 2022-06-06 — inflation/Fed shock (model's momentum features
                      were still bullish into a Fed-driven rotation)
    3. 2024-04-12 — AI/geopolitical pullback (concentrated megacap
                      tech picks, correlation filter did not fire)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models import FEATURE_COLUMNS


# (date, summary_label, context_one_liner, lesson_one_liner)
CASES: list[dict] = [
    {
        "date": "2020-02-18",
        "label": "Pre-COVID risk-on lock-in",
        "context": (
            "SPY was within 2% of all-time highs. The 200dMA regime gate "
            "registered risk_on; VIX percentile was modest. Three trading "
            "days later the market began its fastest 30% drawdown in history."
        ),
        "lesson": (
            "The regime gate is binary and 200dMA-lagged. A volatility "
            "expansion that hasn't yet been realized doesn't reduce sizing. "
            "An IV-based or breadth-based regime input would have caught "
            "this earlier."
        ),
    },
    {
        "date": "2022-06-06",
        "label": "Inflation-print whipsaw",
        "context": (
            "The week's CPI print came in hot on 6/10; SPY fell ~6% in five "
            "days. Both picks were cyclicals (AMZN, JPM) — exactly the "
            "names momentum features were still pointing at coming off Q1 "
            "bounces. The basket return was -8.99%."
        ),
        "lesson": (
            "5-day forward returns are dominated by single macro events. "
            "An earnings/event-aware feature (no entry into the days "
            "before a known print) would not catch CPI directly, but the "
            "logic generalizes: do not trade through known catalysts the "
            "model cannot see."
        ),
    },
    {
        "date": "2024-04-12",
        "label": "Megacap-tech concentration pullback",
        "context": (
            "Iran-Israel headlines on 4/13 + a hotter-than-expected CPI on "
            "4/10 unwound the AI trade. Both picks were META and NVDA — "
            "60-day correlation between them sat just under the 0.80 "
            "drop-the-second-pick threshold, so both stayed in the basket."
        ),
        "lesson": (
            "Correlation thresholding is too coarse: 0.79 = full pair, "
            "0.81 = drop. A continuous co-vol penalty (scale weight down "
            "as correlation rises) or a sector cap would handle this "
            "edge case more honestly."
        ),
    },
]


def build_case_studies() -> dict:
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    preds = pd.read_parquet(results_dir / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])

    v2_trades = pd.read_parquet(results_dir / "v2_trades.parquet")
    v2_trades["date"] = pd.to_datetime(v2_trades["date"])

    shap_df = pd.read_parquet(results_dir / "shap_values.parquet")
    shap_df["date"] = pd.to_datetime(shap_df["date"])

    case_records: list[dict] = []

    for case in CASES:
        case_date = pd.Timestamp(case["date"])

        # Trade record from v2_trades
        trade_row = v2_trades[v2_trades["date"] == case_date]
        if trade_row.empty:
            print(f"[warn] no v2 trade on {case['date']}, skipping")
            continue
        tr = trade_row.iloc[0]
        tickers = [t.strip() for t in tr["picks_csv"].split(",") if t.strip()]

        # Per-pick details: y_proba + realized fwd return + SHAP top contributors
        pick_details: list[dict] = []
        for ticker in tickers:
            pred_row = preds[(preds["date"] == case_date) & (preds["ticker"] == ticker)]
            if pred_row.empty:
                pick_details.append({
                    "ticker": ticker,
                    "y_proba": None,
                    "fwd_return_5d": None,
                    "top_contributors": [],
                })
                continue
            pr = pred_row.iloc[0]

            # SHAP for this pick on this date
            sh = shap_df[(shap_df["date"] == case_date) & (shap_df["ticker"] == ticker)]
            top_contribs: list[dict] = []
            base_value: float | None = None
            if not sh.empty:
                sh_row = sh.iloc[0]
                base_value = float(sh_row["base_value"])
                contribs = sh_row[FEATURE_COLUMNS].astype(float)
                # Sort by absolute magnitude, keep top 5
                order = contribs.abs().sort_values(ascending=False).index
                for feat in order[:5]:
                    top_contribs.append({
                        "feature": feat,
                        "shap_value": float(contribs[feat]),
                    })

            pick_details.append({
                "ticker": ticker,
                "y_proba": float(pr["y_proba"]),
                "fwd_return_5d": float(pr["fwd_return_5d"]) if pd.notna(pr["fwd_return_5d"]) else None,
                "base_value": base_value,
                "top_contributors": top_contribs,
            })

        case_records.append({
            "date": case["date"],
            "label": case["label"],
            "context": case["context"],
            "lesson": case["lesson"],
            "basket_return_net": float(tr["basket_return_net"]),
            "gross_weight": float(tr["gross_weight"]),
            "regime": str(tr["regime"]),
            "picks": pick_details,
        })

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_cases": len(case_records),
        "feature_columns": FEATURE_COLUMNS,
        "cases": case_records,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    out = build_case_studies()

    path = results_dir / "failure_modes.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"=== Failure-mode case studies ({out['n_cases']}) ===")
    for c in out["cases"]:
        print(f"\n  {c['date']}  {c['label']}")
        print(f"    basket: {c['basket_return_net']*100:+.2f}%, regime: {c['regime']}, gross: {c['gross_weight']:.3f}")
        for p in c["picks"]:
            ret = f"{p['fwd_return_5d']*100:+.2f}%" if p["fwd_return_5d"] is not None else "n/a"
            print(f"    {p['ticker']}: y_proba={p['y_proba']:.3f}, fwd_5d={ret}")
            for tc in p["top_contributors"][:3]:
                sign = "+" if tc["shap_value"] >= 0 else ""
                print(f"      {tc['feature']:22s} {sign}{tc['shap_value']:.4f}")
    print(f"\nSaved: {path}")
