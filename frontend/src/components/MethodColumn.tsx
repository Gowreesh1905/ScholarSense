import type { MethodResult } from "../lib/api";
import { methodColor, tint } from "../lib/methodStyles";
import { ResultCard } from "./ResultCard";

interface MethodColumnProps {
  method: MethodResult;
  overlapLabels: (docIndex: number) => string[];
}

function formatMs(ms: number): string {
  return ms < 10 ? `${ms.toFixed(1)} ms` : `${Math.round(ms)} ms`;
}

export function MethodColumn({ method, overlapLabels }: MethodColumnProps) {
  const color = methodColor(method);
  const scores = method.results.map((h) => h.score);
  const range: [number, number] = scores.length ? [Math.min(...scores), Math.max(...scores)] : [0, 1];

  return (
    <section className="flex min-w-0 flex-1 flex-col">
      <div
        className="mb-3 rounded-xl border px-4 py-3"
        style={{ borderColor: tint(color, 0x40), backgroundColor: tint(color, 0x14) }}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-baseline gap-2">
            <span className="h-2 w-2 shrink-0 -translate-y-px rounded-full" style={{ backgroundColor: color }} />
            <h2 className="font-display text-[15px] font-semibold leading-snug text-ink">{method.label}</h2>
          </div>
          <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-faint" title="Time this method took for the query">
            {formatMs(method.took_ms)}
          </span>
        </div>
        <p className="mt-0.5 text-[11px] leading-snug text-ink-muted">{method.description}</p>
      </div>

      <div className="flex flex-col gap-3">
        {method.results.length === 0 ? (
          <p className="rounded-xl border border-dashed border-line px-4 py-6 text-center text-xs text-ink-faint">
            No results: none of the query's words are in this method's vocabulary.
          </p>
        ) : (
          method.results.map((hit, i) => (
            <ResultCard
              key={hit.doc_index}
              hit={hit}
              color={color}
              scoreType={method.score_type}
              scoreRange={range}
              index={i}
              foundByOtherMethods={overlapLabels(hit.doc_index)}
            />
          ))
        )}
      </div>
    </section>
  );
}
