"use client";

import EvaluationPanel from "../components/EvaluationPanel";
import TopNav from "../components/TopNav";

export default function EvaluationPage() {
  return (
    <div style={{ height: "calc(100vh - 22px)", width: "100vw", display: "flex", flexDirection: "column", background: "var(--bg-base)" }}>
      <div style={{ background: "var(--bg-panel)", borderBottom: "1px solid var(--border)", padding: "8px 16px", display: "flex", alignItems: "center", gap: 16 }}>
        <TopNav variant="page" />
        <span style={{ fontSize: 12, fontWeight: 700, letterSpacing: "-0.02em", marginLeft: "auto" }}>
          <span style={{ background: "linear-gradient(90deg, var(--cyan), var(--green))", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
            Model Evaluation
          </span>
        </span>
      </div>
      <div style={{ flex: 1, overflow: "hidden" }}>
        <EvaluationPanel />
      </div>
    </div>
  );
}
