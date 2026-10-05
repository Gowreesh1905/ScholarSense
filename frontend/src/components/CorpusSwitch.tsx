import type { CorpusInfo } from "../lib/api";

interface CorpusSwitchProps {
  corpora: CorpusInfo[];
  value: string;
  onChange: (corpus: string) => void;
  disabled?: boolean;
}

const HINTS: Record<string, string> = {
  "727": "The Scholar Inbox papers: every method, plus aspect search.",
  scale: "The same papers plus arXiv computer-science abstracts, indexed by the laptop cluster.",
};

export function CorpusSwitch({ corpora, value, onChange, disabled }: CorpusSwitchProps) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        role="radiogroup"
        aria-label="Which collection of papers to search"
        className="flex max-w-full flex-wrap justify-center gap-1 rounded-xl border border-line bg-surface p-1 shadow-soft"
      >
        {corpora.map((c) => {
          const active = c.key === value;
          return (
            <button
              key={c.key}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(c.key)}
              className={`rounded-lg px-3.5 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${
                active ? "bg-accent text-white" : "text-ink-muted hover:text-ink"
              }`}
            >
              {c.label}
            </button>
          );
        })}
      </div>
      <p className="text-center text-[11px] text-ink-faint">{HINTS[value] ?? ""}</p>
    </div>
  );
}
