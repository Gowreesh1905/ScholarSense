import { useTheme } from "../hooks/useTheme";
import type { Route } from "../hooks/useHashRoute";
import type { HealthResponse } from "../lib/api";

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" className="h-4 w-4">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z" />
    </svg>
  );
}

const TABS: { route: Route; label: string }[] = [
  { route: "search", label: "Search" },
  { route: "results", label: "Results" },
];

interface HeaderProps {
  health: HealthResponse | null;
  healthError: boolean;
  route: Route;
}

export function Header({ health, healthError, route }: HeaderProps) {
  const { theme, toggle } = useTheme();

  return (
    <header className="sticky top-0 z-20 border-b border-line/80 bg-canvas/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 py-3 sm:gap-4 sm:px-8 sm:py-3.5">
        <div className="flex min-w-0 items-center gap-3 sm:gap-6">
          <a href="#search" className="flex shrink-0 items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white shadow-soft">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="h-4 w-4">
                <circle cx="10.5" cy="10.5" r="6.5" />
                <path d="M20 20l-4.8-4.8" />
              </svg>
            </div>
            <div className="hidden leading-tight sm:block">
              <p className="font-display text-[17px] font-semibold tracking-tight text-ink">ScholarSense</p>
              <p className="text-[11px] text-ink-faint">Comparative semantic search</p>
            </div>
          </a>

          <nav className="flex items-center gap-1 rounded-full border border-line bg-surface p-0.5">
            {TABS.map((t) => (
              <a
                key={t.route}
                href={`#${t.route}`}
                aria-current={route === t.route ? "page" : undefined}
                className={`rounded-full px-3 py-1 text-xs font-medium transition sm:px-3.5 ${
                  route === t.route ? "bg-accent text-white" : "text-ink-muted hover:text-ink"
                }`}
              >
                {t.label}
              </a>
            ))}
          </nav>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="hidden items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-ink-muted md:flex">
            <span className={`h-1.5 w-1.5 rounded-full ${healthError ? "bg-red-500" : health ? "bg-emerald-500" : "bg-amber-500 animate-pulse"}`} />
            {healthError
              ? "Backend offline"
              : health
                ? `${health.corpus_size.toLocaleString()} papers · ${health.methods.length} methods · ${health.device.toUpperCase()}`
                : "Connecting…"}
          </div>
          <button
            type="button"
            onClick={toggle}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-line bg-surface text-ink-muted transition hover:border-line-strong hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
          >
            {theme === "dark" ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </div>
    </header>
  );
}
