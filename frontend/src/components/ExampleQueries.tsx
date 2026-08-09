const EXAMPLES = [
  "automatic translation of spoken language",
  "teaching robots to grasp unfamiliar objects",
  "detecting fake news with graph structure",
  "compressing large language models",
  "self-supervised representation learning for images",
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
