// Single source of truth for the order safety caps.
// MUST match dashboard/backend/main.py:
//   MAX_QTY_PER_ORDER = 100
//   MAX_NOTIONAL_PER_ORDER = 10_000

export const MAX_QTY_PER_ORDER = 100;
export const MAX_NOTIONAL_PER_ORDER = 10_000;

/**
 * Given a recommended share count and current reference price, return the
 * largest qty that respects BOTH safety caps (per-order share cap +
 * per-order notional cap). Also reports whether the cap is binding so
 * the UI can flag "(capped)" honestly rather than silently down-sizing.
 */
export function cappedTradableShares(
  recShares: number | null | undefined,
  lastClose: number | null | undefined,
): { shares: number; isCapped: boolean; reason: string | null } {
  const rec = Math.max(0, Math.floor(recShares ?? 0));
  if (rec < 1) return { shares: 0, isCapped: false, reason: null };
  if (!lastClose || lastClose <= 0) return { shares: rec, isCapped: false, reason: null };

  const notionalCap = Math.floor(MAX_NOTIONAL_PER_ORDER / lastClose);
  const qtyCap = MAX_QTY_PER_ORDER;
  const cap = Math.min(rec, notionalCap, qtyCap);

  if (cap === rec) {
    return { shares: rec, isCapped: false, reason: null };
  }
  const binding = cap === notionalCap ? `$${MAX_NOTIONAL_PER_ORDER.toLocaleString()} notional cap` : `${MAX_QTY_PER_ORDER} share cap`;
  return { shares: Math.max(0, cap), isCapped: true, reason: binding };
}
