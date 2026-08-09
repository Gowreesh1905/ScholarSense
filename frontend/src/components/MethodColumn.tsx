import type { MethodResult } from "../lib/api";
import { METHOD_STYLES, METHOD_TAGLINE } from "../lib/methodStyles";
import { ResultCard } from "./ResultCard";

interface MethodColumnProps {
  method: MethodResult;
  overlapLabels: (arxivId: string) => string[];
}

export function MethodColumn({ method, overlapLabels }: MethodColumnProps) {
  const style = METHOD_STYLES[method.key];

  return (
    <section className="flex min-w-0 flex-1 flex-col">
      <div className={`mb-3 rounded-xl border ${style.border} ${style.bgSoft} px-4 py-3`}>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
          <h2 className="font-display text-[15px] font-semibold text-ink">{method.label}</h2>
        </div>
        <p className="mt-0.5 text-[11px] text-ink-muted">{METHOD_TAGLINE[method.key]}</p>
      </div>

      <div className="flex flex-col gap-3">
        {method.results.length === 0 ? (
          <p className="rounded-xl border border-dashed border-line px-4 py-6 text-center text-xs text-ink-faint">
            No results.
          </p>
        ) : (
          method.results.map((hit, i) => (
            <ResultCard
              key={hit.arxiv_id + hit.rank}
              hit={hit}
              method={method.key}
              index={i}
              foundByOtherMethods={overlapLabels(hit.arxiv_id)}
            />
          ))
        )}
      </div>
    </section>
  );
}
