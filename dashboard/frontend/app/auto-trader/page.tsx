"use client";

import AutoTraderPanel from "../components/AutoTraderPanel";
import TopNav from "../components/TopNav";

export default function AutoTraderPage() {
  return (
    <div style={{ height: "100vh", width: "100vw", display: "flex", flexDirection: "column", background: "var(--bg-base)" }}>
      <div style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "8px 16px", display: "flex", alignItems: "center", gap: 16 }}>
        <TopNav variant="page" />
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "-0.02em", marginLeft: "auto" }}>
          <span style={{ background: "linear-gradient(90deg, var(--green), var(--cyan))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            Auto-Trader
          </span>
        </span>
      </div>
      <div style={{ flex: 1, overflow: "hidden" }}>
        <AutoTraderPanel />
      </div>
    </div>
  );
}
