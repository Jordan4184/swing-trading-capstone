"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const PAGES: { href: string; label: string }[] = [
  { href: "/", label: "Dashboard" },
  { href: "/journal", label: "Journal" },
  { href: "/auto-trader", label: "Auto-Trader" },
  { href: "/evaluation", label: "Evaluation" },
];

type Variant = "compact" | "page";

export default function TopNav({ variant = "compact" }: { variant?: Variant }) {
  const pathname = usePathname();

  const linkBase: React.CSSProperties = variant === "compact"
    ? {
        fontSize: 11,
        fontWeight: 600,
        textDecoration: "none",
        padding: "3px 10px",
        borderRadius: 3,
        letterSpacing: "0.02em",
        whiteSpace: "nowrap",
      }
    : {
        fontSize: 12,
        fontWeight: 600,
        textDecoration: "none",
        padding: "4px 12px",
        borderRadius: 4,
        letterSpacing: "0.02em",
        whiteSpace: "nowrap",
      };

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname?.startsWith(href);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      {PAGES.map((p) => {
        const active = isActive(p.href);
        return (
          <Link
            key={p.href}
            href={p.href}
            style={{
              ...linkBase,
              background: active ? "var(--green-bg)" : "transparent",
              color: active ? "var(--green)" : "var(--text-muted)",
              border: active ? "1px solid rgba(74, 222, 128, 0.3)" : "1px solid transparent",
            }}
          >
            {p.label}
          </Link>
        );
      })}
    </div>
  );
}
