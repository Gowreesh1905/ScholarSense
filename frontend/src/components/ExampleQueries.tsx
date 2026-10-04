// Plain-language queries on topics the 727-paper corpus actually covers (mostly computer
// vision), worded without the papers' own jargon so vocabulary mismatch shows up.
const EXAMPLES = [
  "making blurry photos sharp again",
  "turning a single photo into a 3D model",
  "self-driving cars sensing surroundings with laser scanners",
  "teaching robots to grasp unfamiliar objects",
  "learning image features without human labels",
];

interface ExampleQueriesProps {
  onPick: (query: string) => void;
  disabled: boolean;
}

export function ExampleQueries({ onPick, disabled }: ExampleQueriesProps) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      <span className="text-xs text-ink-faint">Try:</span>
      {EXAMPLES.map((q) => (
        <button
          key={q}
          type="button"
          disabled={disabled}
          onClick={() => onPick(q)}
          className="rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-ink-muted transition hover:border-accent/40 hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
        >
          {q}
        </button>
      ))}
    </div>
  );
}
