"""
Autonomous trading scheduler.

Runs the entry/exit cycles on weekday market hours via APScheduler.
Each cycle logs to the runs table so you can audit what happened.

Entry cycle (weekdays 9:35am ET):
  1. Read latest predictions from the model
  2. Ask position_manager which signals to trade
  3. For each approved signal: size, record intent, submit to Alpaca
  4. Log run with success/failure counts

Exit cycle (weekdays 3:55pm ET):
  1. Find positions whose target_exit_date is today or earlier
  2. Submit market sell orders for each
  3. Record exit fills
  4. Log run

Manual trigger endpoints (for testing) live in main.py.
The scheduler is OPT-IN via env var AUTO_TRADER_ENABLED=true.
"""

import os
from datetime import datetime, date
from typing import Optional

import auto_trader_db as db
import position_manager as pm

# These imports happen inside functions so the module can be imported
# in test contexts without Alpaca credentials configured.

# Safety caps - cannot be overridden by callers
MAX_ORDERS_PER_RUN = 5
MAX_EXITS_PER_RUN = 10


def _get_alpaca_clients():
    """Lazy import to avoid hard dep at module load time."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Alpaca credentials missing - cannot run scheduler")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    from alpaca.trading.client import TradingClient

    data_client = StockHistoricalDataClient(api_key, api_secret)
    trading_client = TradingClient(api_key, api_secret, paper=True)
    return data_client, trading_client, StockLatestQuoteRequest


def _load_latest_predictions(top_n: int = 10) -> list:
    """Load the latest predictions from disk - same logic as main.py."""
    from pathlib import Path
    import pandas as pd

    backend_dir = Path(__file__).parent
    project_root = backend_dir.parent.parent
    pred_path = project_root / "results" / "predictions.parquet"

    df = pd.read_parquet(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest = (
        df[df["date"] == latest_date]
        .sort_values("y_proba", ascending=False)
        .head(top_n)
        .copy()
    )

    return [
        {
            "ticker": row["ticker"],
            "y_proba": float(row["y_proba"]),
            "signal_date": latest_date.strftime("%Y-%m-%d"),
        }
        for _, row in latest.iterrows()
    ]


def _get_quote(data_client, request_class, ticker: str) -> dict:
    """Fetch latest quote for sizing."""
    req = request_class(symbol_or_symbols=[ticker])
    resp = data_client.get_stock_latest_quote(req)
    if ticker not in resp:
        return {"bid": None, "ask": None, "mid": None}
    q = resp[ticker]
    bid = float(q.bid_price) if q.bid_price else None
    ask = float(q.ask_price) if q.ask_price else None
    mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
    return {"bid": bid, "ask": ask, "mid": mid}


def run_entry_cycle(dry_run: bool = False) -> dict:
    """
    Execute the daily entry cycle.

    Args:
        dry_run: If True, only computes which signals WOULD be traded
                 without placing orders or recording intents.

    Returns:
        Summary dict with counts and details.
    """
    summary = {
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "signals_evaluated": 0,
        "signals_selected": 0,
        "orders_attempted": 0,
        "orders_succeeded": 0,
        "orders_failed": 0,
        "trades": [],
        "errors": [],
    }

    try:
        # Load predictions
        predictions = _load_latest_predictions(top_n=10)
        summary["signals_evaluated"] = len(predictions)

        # Set up Alpaca clients early so we can build the risk context
        data_client, trading_client, StockLatestQuoteRequest = _get_alpaca_clients()

        # Build risk context (Alpaca daily bars for the universe + cached VIX).
        # If this fails for any reason, select_signals_to_trade falls back to
        # the flat-20% weighting — sizing degrades gracefully, no crash.
        universe_tickers = sorted(set(p["ticker"] for p in predictions) | {"SPY"})
        risk_context = pm.build_risk_context(data_client, universe_tickers)

        selected, risk_diag = pm.select_signals_to_trade(
            predictions,
            max_new=MAX_ORDERS_PER_RUN,
            risk_context=risk_context,
        )
        summary["signals_selected"] = len(selected)
        summary["risk"] = risk_diag

        if dry_run:
            summary["trades"] = [
                {
                    "ticker": p["ticker"],
                    "y_proba": p["y_proba"],
                    "weight": round(w, 4),
                    "would_trade": True,
                }
                for p, w in selected
            ]
            db.log_run(
                run_type="entry_dryrun",
                n_signals_evaluated=summary["signals_evaluated"],
                n_orders_placed=0,
                notes=f"dry run [{risk_diag.get('regime')}], would trade {len(selected)}",
            )
            summary["completed_at"] = datetime.now().isoformat()
            return summary

        if not selected:
            db.log_run(
                run_type="entry",
                n_signals_evaluated=summary["signals_evaluated"],
                n_orders_placed=0,
                notes=f"no signals met criteria [{risk_diag.get('regime')}]",
            )
            summary["completed_at"] = datetime.now().isoformat()
            return summary

        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Get account for sizing
        account = trading_client.get_account()
        buying_power = float(account.buying_power)

        # Place orders
        for pred, weight in selected:
            ticker = pred["ticker"]
            signal_proba = pred["y_proba"]
            signal_date = pred["signal_date"]
            summary["orders_attempted"] += 1

            try:
                # Get current quote for sizing
                quote = _get_quote(data_client, StockLatestQuoteRequest, ticker)
                if not quote.get("ask") and not quote.get("mid"):
                    summary["orders_failed"] += 1
                    summary["errors"].append(f"{ticker}: no quote available")
                    continue

                ref_price = quote["ask"] or quote["mid"]
                qty = pm.calculate_position_size(buying_power, ref_price, weight=weight)
                if qty < 1:
                    summary["orders_failed"] += 1
                    summary["errors"].append(f"{ticker}: sized to 0 shares")
                    continue

                # Find rank for this ticker in the predictions
                rank = next((i + 1 for i, p in enumerate(predictions) if p["ticker"] == ticker), 99)

                # Record intent FIRST so we don't double-place if Alpaca call fails
                trade_id = db.record_intent(
                    ticker=ticker,
                    signal_date=signal_date,
                    signal_proba=signal_proba,
                    signal_rank=rank,
                    target_holding_days=pm.DEFAULT_HOLDING_DAYS,
                    qty=qty,
                )
                if trade_id is None:
                    summary["errors"].append(f"{ticker}: duplicate intent (already recorded today)")
                    continue

                # Submit to Alpaca
                order_req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = trading_client.submit_order(order_data=order_req)
                db.update_submission(
                    trade_id=trade_id,
                    alpaca_order_id=str(order.id),
                    submission_status="submitted",
                )
                summary["orders_succeeded"] += 1
                summary["trades"].append({
                    "ticker": ticker,
                    "qty": qty,
                    "ref_price": ref_price,
                    "estimated_notional": round(ref_price * qty, 2),
                    "trade_id": trade_id,
                    "alpaca_order_id": str(order.id),
                })
            except Exception as e:
                summary["orders_failed"] += 1
                err_msg = f"{ticker}: {type(e).__name__}: {str(e)}"
                summary["errors"].append(err_msg)
                # Try to update the trade record if we got that far
                try:
                    db.update_submission(
                        trade_id=trade_id,
                        alpaca_order_id=None,
                        submission_status="error",
                        submission_error=str(e),
                    )
                except Exception:
                    pass

        db.log_run(
            run_type="entry",
            n_signals_evaluated=summary["signals_evaluated"],
            n_orders_placed=summary["orders_succeeded"],
            notes=f"placed {summary['orders_succeeded']}/{summary['orders_attempted']}",
            error="; ".join(summary["errors"]) if summary["errors"] else None,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        summary["errors"].append(f"FATAL: {err}")
        db.log_run(run_type="entry", error=err)

    summary["completed_at"] = datetime.now().isoformat()
    return summary


def run_exit_cycle(dry_run: bool = False) -> dict:
    """
    Execute the daily exit cycle.

    For each position whose target_exit_date <= today:
      Submit market sell order, record exit.
    """
    summary = {
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "positions_due": 0,
        "exits_attempted": 0,
        "exits_succeeded": 0,
        "exits_failed": 0,
        "exits": [],
        "errors": [],
    }

    try:
        positions = pm.get_exits_due()
        summary["positions_due"] = len(positions)

        if dry_run:
            summary["exits"] = [
                {"ticker": p["ticker"], "qty": p["qty"], "would_exit": True}
                for p in positions[:MAX_EXITS_PER_RUN]
            ]
            db.log_run(
                run_type="exit_dryrun",
                n_positions_exited=0,
                notes=f"dry run, {len(positions)} positions due",
            )
            summary["completed_at"] = datetime.now().isoformat()
            return summary

        if not positions:
            db.log_run(run_type="exit", n_positions_exited=0, notes="no positions due")
            summary["completed_at"] = datetime.now().isoformat()
            return summary

        data_client, trading_client, StockLatestQuoteRequest = _get_alpaca_clients()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        for pos in positions[:MAX_EXITS_PER_RUN]:
            summary["exits_attempted"] += 1
            ticker = pos["ticker"]
            qty = pos["qty"]
            trade_id = pos["id"]

            try:
                order_req = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                order = trading_client.submit_order(order_data=order_req)

                # Get a current quote to estimate exit price (actual fill comes later via sync)
                quote = _get_quote(data_client, StockLatestQuoteRequest, ticker)
                exit_price = quote.get("bid") or quote.get("mid") or pos["entry_price"]

                db.record_exit(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_filled_at=datetime.now().isoformat(),
                    exit_alpaca_order_id=str(order.id),
                    exit_reason="scheduled",
                )
                summary["exits_succeeded"] += 1
                summary["exits"].append({
                    "ticker": ticker,
                    "qty": qty,
                    "exit_price": exit_price,
                    "alpaca_order_id": str(order.id),
                })
            except Exception as e:
                summary["exits_failed"] += 1
                summary["errors"].append(f"{ticker}: {type(e).__name__}: {str(e)}")

        db.log_run(
            run_type="exit",
            n_positions_exited=summary["exits_succeeded"],
            notes=f"exited {summary['exits_succeeded']}/{summary['exits_attempted']}",
            error="; ".join(summary["errors"]) if summary["errors"] else None,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        summary["errors"].append(f"FATAL: {err}")
        db.log_run(run_type="exit", error=err)

    summary["completed_at"] = datetime.now().isoformat()
    return summary




# ---------------------------------------------------------------------------
# Prediction cycle: invokes the capstone venv to regenerate predictions.parquet
# ---------------------------------------------------------------------------

import subprocess
from pathlib import Path

# Path to the capstone venv's Python (where yfinance/sklearn/joblib live)
_BACKEND_DIR = Path(__file__).parent
_PROJECT_ROOT = _BACKEND_DIR.parent.parent
_CAPSTONE_PYTHON = _PROJECT_ROOT / "venv" / "bin" / "python3"


def run_prediction_cycle(timeout_seconds: int = 120) -> dict:
    """
    Run the daily prediction generation by invoking the capstone venv.

    This shells out to the capstone\'s Python interpreter (which has yfinance,
    sklearn, joblib, etc.) and runs `python -m src.predict`. The script
    fetches fresh market data, runs inference with the saved model, and
    appends today\'s predictions to results/predictions.parquet.
    """
    summary = {
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "success": False,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "errors": [],
    }

    if not _CAPSTONE_PYTHON.exists():
        err = f"Capstone Python not found at {_CAPSTONE_PYTHON}"
        summary["errors"].append(err)
        db.log_run(run_type="predict", error=err)
        summary["completed_at"] = datetime.now().isoformat()
        return summary

    try:
        result = subprocess.run(
            [str(_CAPSTONE_PYTHON), "-m", "src.predict"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        summary["stdout"] = result.stdout[-2000:]  # last 2000 chars to keep it manageable
        summary["stderr"] = result.stderr[-2000:]
        summary["returncode"] = result.returncode
        summary["success"] = result.returncode == 0

        if result.returncode != 0:
            summary["errors"].append(f"Subprocess exited with code {result.returncode}")

        notes = f"returncode={result.returncode}, stdout_lines={len(result.stdout.splitlines())}"
        db.log_run(
            run_type="predict",
            notes=notes,
            error="; ".join(summary["errors"]) if summary["errors"] else None,
        )
    except subprocess.TimeoutExpired:
        err = f"Prediction subprocess timed out after {timeout_seconds}s"
        summary["errors"].append(err)
        db.log_run(run_type="predict", error=err)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        summary["errors"].append(err)
        db.log_run(run_type="predict", error=err)

    summary["completed_at"] = datetime.now().isoformat()
    return summary


# ---------------------------------------------------------------------------
# APScheduler setup
# ---------------------------------------------------------------------------

_scheduler = None


def get_scheduler_status() -> dict:
    """Return scheduler state."""
    if _scheduler is None:
        return {"running": False, "enabled_via_env": os.getenv("AUTO_TRADER_ENABLED", "false") == "true"}
    jobs = _scheduler.get_jobs()
    return {
        "running": _scheduler.running,
        "enabled_via_env": os.getenv("AUTO_TRADER_ENABLED", "false") == "true",
        "jobs": [
            {
                "id": j.id,
                "name": j.name,
                "next_run_time": str(j.next_run_time) if j.next_run_time else None,
            }
            for j in jobs
        ],
    }


def start_scheduler() -> dict:
    """Start the APScheduler if AUTO_TRADER_ENABLED env var is true. Idempotent."""
    global _scheduler

    if os.getenv("AUTO_TRADER_ENABLED", "false") != "true":
        return {"started": False, "reason": "AUTO_TRADER_ENABLED not set to 'true'"}

    if _scheduler is not None and _scheduler.running:
        return {"started": True, "reason": "already running"}

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    import pytz

    et = pytz.timezone("America/New_York")

    _scheduler = BackgroundScheduler(timezone=et)

    # Entry: weekdays 9:35am ET
    _scheduler.add_job(
        run_entry_cycle,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone=et),
        id="entry_cycle",
        name="Daily entry cycle",
        replace_existing=True,
        max_instances=1,
    )

    # Predict: weekdays 9:30am ET (5 min before entry)
    _scheduler.add_job(
        run_prediction_cycle,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=30, timezone=et),
        id="predict_cycle",
        name="Daily prediction generation",
        replace_existing=True,
        max_instances=1,
    )

    # Exit: weekdays 3:55pm ET
    _scheduler.add_job(
        run_exit_cycle,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=55, timezone=et),
        id="exit_cycle",
        name="Daily exit cycle",
        replace_existing=True,
        max_instances=1,
    )

    _scheduler.start()
    return {"started": True, "reason": "scheduler running"}


def stop_scheduler() -> dict:
    """Stop the scheduler if running."""
    global _scheduler
    if _scheduler is None or not _scheduler.running:
        return {"stopped": False, "reason": "not running"}
    _scheduler.shutdown(wait=False)
    return {"stopped": True}
