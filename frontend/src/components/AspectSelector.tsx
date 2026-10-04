import { methodColor } from "../lib/methodStyles";

const ASPECT_LABELS: Record<string, string> = {
  all: "Whole abstract",
  task: "Task",
  problem: "Problem",
  method: "Method",
  result: "Result",
};

const ASPECT_HINTS: Record<string, string> = {
  all: "Every method searches the full abstract.",
  task: "Also search only the task each paper sets out to do.",
  problem: "Also search only the problem each paper says it addresses.",
  method: "Also search only the method or idea each paper proposes.",
  result: "Also search only the results each paper reports.",
};

interface AspectSelectorProps {
  aspects: string[];
  coverage: Record<string, number>;
  value: string;
  onChange: (aspect: string) => void;
  disabled?: boolean;
}

export function AspectSelector({ aspects, coverage, value, onChange, disabled }: AspectSelectorProps) {
  const color = methodColor({ key: "aspect" });

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        role="radiogroup"
        aria-label="Which part of the abstract to search"
        className="flex max-w-full flex-wrap justify-center gap-1 rounded-xl border border-line bg-surface p-1 shadow-soft"
      >
        {aspects.map((aspect) => {
          const active = aspect === value;
          const count = coverage[aspect];
          return (
            <button
              key={aspect}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(aspect)}
              className="rounded-lg px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-50"
              style={active ? { backgroundColor: aspect === "all" ? "var(--surface-hover)" : color, color: aspect === "all" ? "var(--ink)" : "#fff" } : { color: "var(--ink-muted)" }}
            >
              {ASPECT_LABELS[aspect] ?? aspect}
              {count !== undefined && (
                <span className={active && aspect !== "all" ? "ml-1 opacity-80" : "ml-1 text-ink-faint"}>· {count}</span>
              )}
            </button>
          );
        })}
      </div>
      <p className="text-[11px] text-ink-faint">
        {ASPECT_HINTS[value] ?? ""}
        {value !== "all" && " The matched sentence is highlighted."}
      </p>
    </div>
  );
}
