import type { MethodKey } from "../lib/api";
import { METHOD_STYLES, METHOD_TAGLINE } from "../lib/methodStyles";

const LABELS: Record<MethodKey, string> = {
  tfidf: "TF-IDF",
  word2vec: "Word2Vec",
  bert: "BERT / SPECTER",
};

interface LoadingColumnProps {
  method: MethodKey;
  count?: number;
}

export function LoadingColumn({ method, count = 5 }: LoadingColumnProps) {
  const style = METHOD_STYLES[method];

  return (
    <section className="flex min-w-0 flex-1 flex-col">
      <div className={`mb-3 rounded-xl border ${style.border} ${style.bgSoft} px-4 py-3`}>
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${style.dot}`} />
          <h2 className="font-display text-[15px] font-semibold text-ink">{LABELS[method]}</h2>
        </div>
        <p className="mt-0.5 text-[11px] text-ink-muted">{METHOD_TAGLINE[method]}</p>
      </div>

      <div className="flex flex-col gap-3">
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="rounded-xl border border-line bg-surface p-4 shadow-soft">
            <div className="mb-3 flex items-center gap-2">
              <div className="skeleton h-5 w-5 rounded-full" />
              <div className="skeleton h-3 w-24 rounded" />
            </div>
            <div className="mb-1.5 skeleton h-2.5 w-full rounded" />
            <div className="mb-1.5 skeleton h-2.5 w-full rounded" />
            <div className="mb-3 skeleton h-2.5 w-2/3 rounded" />
            <div className="skeleton h-1.5 w-full rounded-full" />
          </div>
        ))}
      </div>
    </section>
  );
}
