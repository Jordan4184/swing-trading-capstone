import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "auto_trader.db"

SCHEMA_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    signal_proba REAL NOT NULL,
    signal_rank INTEGER NOT NULL,
    target_holding_days INTEGER NOT NULL DEFAULT 5,
    intent_at TEXT NOT NULL,
    alpaca_order_id TEXT,
    submission_status TEXT NOT NULL,
    submission_error TEXT,
    entry_price REAL,
    entry_filled_at TEXT,
    qty INTEGER,
    target_exit_date TEXT,
    exit_price REAL,
    exit_filled_at TEXT,
    exit_alpaca_order_id TEXT,
    exit_reason TEXT,
    pnl REAL,
    pnl_pct REAL,
    model_version TEXT NOT NULL DEFAULT 'rf_v1',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, signal_date, model_version)
);
"""

SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    n_signals_evaluated INTEGER,
    n_orders_placed INTEGER,
    n_positions_exited INTEGER,
    notes TEXT,
    error TEXT
);
"""

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_trades_signal_date ON trades(signal_date);",
]


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA_TRADES)
        conn.execute(SCHEMA_RUNS)
        for stmt in INDEX_STATEMENTS:
            conn.execute(stmt)


def record_intent(ticker, signal_date, signal_proba, signal_rank, target_holding_days=5, qty=1, model_version="rf_v1"):
    target_exit = (datetime.strptime(signal_date, "%Y-%m-%d").date() + timedelta(days=target_holding_days)).isoformat()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM trades WHERE ticker = ? AND signal_date = ? AND model_version = ?",
            (ticker, signal_date, model_version),
        ).fetchone()
        if existing:
            return None
        cursor = conn.execute(
            "INSERT INTO trades (ticker, signal_date, signal_proba, signal_rank, target_holding_days, intent_at, submission_status, qty, target_exit_date, model_version) VALUES (?, ?, ?, ?, ?, datetime('now'), 'pending', ?, ?, ?)",
            (ticker, signal_date, signal_proba, signal_rank, target_holding_days, qty, target_exit, model_version),
        )
        return cursor.lastrowid


def update_submission(trade_id, alpaca_order_id, submission_status, submission_error=None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE trades SET alpaca_order_id = ?, submission_status = ?, submission_error = ?, updated_at = datetime('now') WHERE id = ?",
            (alpaca_order_id, submission_status, submission_error, trade_id),
        )


def record_entry_fill(trade_id, entry_price, filled_at, actual_qty):
    with get_conn() as conn:
        conn.execute(
            "UPDATE trades SET entry_price = ?, entry_filled_at = ?, qty = ?, updated_at = datetime('now') WHERE id = ?",
            (entry_price, filled_at, actual_qty, trade_id),
        )


def record_exit(trade_id, exit_price, exit_filled_at, exit_alpaca_order_id, exit_reason="scheduled"):
    with get_conn() as conn:
        row = conn.execute("SELECT entry_price, qty FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if not row or row["entry_price"] is None or row["qty"] is None:
            raise ValueError("Cannot exit trade " + str(trade_id) + ": missing entry data")
        entry_price = row["entry_price"]
        qty = row["qty"]
        pnl = (exit_price - entry_price) * qty
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0
        conn.execute(
            "UPDATE trades SET exit_price = ?, exit_filled_at = ?, exit_alpaca_order_id = ?, exit_reason = ?, pnl = ?, pnl_pct = ?, updated_at = datetime('now') WHERE id = ?",
            (exit_price, exit_filled_at, exit_alpaca_order_id, exit_reason, round(pnl, 4), round(pnl_pct, 6), trade_id),
        )


def get_open_positions():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM trades WHERE entry_filled_at IS NOT NULL AND exit_filled_at IS NULL ORDER BY entry_filled_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_positions_due_for_exit(today_iso=None):
    today_iso = today_iso or date.today().isoformat()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE entry_filled_at IS NOT NULL AND exit_filled_at IS NULL AND target_exit_date <= ? ORDER BY target_exit_date ASC",
            (today_iso,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_pending_orders():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM trades WHERE submission_status = 'submitted' AND entry_filled_at IS NULL ORDER BY intent_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_recent_trades(limit=50):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM trades ORDER BY intent_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_closed_trades(limit=100):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM trades WHERE exit_filled_at IS NOT NULL ORDER BY exit_filled_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_performance_stats():
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as n_trades, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as n_wins, SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as n_losses, COALESCE(SUM(pnl), 0) as total_pnl, COALESCE(AVG(pnl), 0) as avg_pnl, COALESCE(AVG(pnl_pct), 0) as avg_pnl_pct, COALESCE(MAX(pnl), 0) as best_trade, COALESCE(MIN(pnl), 0) as worst_trade FROM trades WHERE exit_filled_at IS NOT NULL"
        ).fetchone()
        n_trades = row["n_trades"] or 0
        n_wins = row["n_wins"] or 0
        hit_rate = n_wins / n_trades if n_trades > 0 else 0.0
        n_open = conn.execute("SELECT COUNT(*) as c FROM trades WHERE entry_filled_at IS NOT NULL AND exit_filled_at IS NULL").fetchone()["c"]
        n_pending = conn.execute("SELECT COUNT(*) as c FROM trades WHERE submission_status = 'submitted' AND entry_filled_at IS NULL").fetchone()["c"]
        return {
            "n_closed_trades": n_trades,
            "n_wins": n_wins,
            "n_losses": row["n_losses"] or 0,
            "hit_rate": round(hit_rate, 4),
            "total_pnl": round(row["total_pnl"], 2),
            "avg_pnl": round(row["avg_pnl"], 2),
            "avg_pnl_pct": round(row["avg_pnl_pct"], 6),
            "best_trade": round(row["best_trade"], 2),
            "worst_trade": round(row["worst_trade"], 2),
            "n_open_positions": n_open,
            "n_pending_orders": n_pending,
        }


def log_run(run_type, n_signals_evaluated=None, n_orders_placed=None, n_positions_exited=None, notes=None, error=None):
    with get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO runs (run_type, n_signals_evaluated, n_orders_placed, n_positions_exited, notes, error) VALUES (?, ?, ?, ?, ?, ?)",
            (run_type, n_signals_evaluated, n_orders_placed, n_positions_exited, notes, error),
        )
        return cursor.lastrowid


def get_recent_runs(limit=20):
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY run_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


init_db()
