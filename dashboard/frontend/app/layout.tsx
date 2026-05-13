import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import HonestyFooter from "./components/HonestyFooter";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Swing Trading Capstone — Jordan Donaldson",
  description: "ML cross-sectional swing trading dashboard with risk-managed v2 strategy, SHAP interpretability, bootstrap CIs, and live paper trading.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col" style={{ paddingBottom: 22 }}>
        {children}
        <HonestyFooter />
      </body>
    </html>
  );
}
