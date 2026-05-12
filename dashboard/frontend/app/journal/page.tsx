"use client";

import JournalPanel from "../components/JournalPanel";
import Link from "next/link";

export default function JournalPage() {
  return (
    <div style={{ height: "100vh", width: "100vw", display: "flex", flexDirection: "column", background: "var(--bg-base)" }}>
      <div style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "8px 16px", display: "flex", alignItems: "center", gap: 16 }}>
        <Link href="/" style={{ color: "var(--text-muted)", fontSize: 11, textDecoration: "none" }}>
          ← Back to Dashboard
        </Link>
        <Link href="/auto-trader" style={{ color: "var(--text-muted)", fontSize: 11, textDecoration: "none" }}>
          Auto-Trader →
        </Link>
        <Link href="/evaluation" style={{ color: "var(--text-muted)", fontSize: 11, textDecoration: "none" }}>
          Evaluation →
        </Link>
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "-0.02em" }}>
          <span style={{ background: "linear-gradient(90deg, var(--cyan), var(--green))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            Trade Journal
          </span>
        </span>
      </div>
      <div style={{ flex: 1, overflow: "hidden" }}>
        <JournalPanel />
      </div>
    </div>
  );
}
