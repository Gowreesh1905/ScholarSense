import type { Family, MethodInfo } from "../lib/api";
import { FAMILY_BLURBS, FAMILY_LABELS, FAMILY_ORDER, methodColor, tint } from "../lib/methodStyles";

interface EmptyStateProps {
  methods: MethodInfo[];
  corpusSize: number | null;
}

export function EmptyState({ methods, corpusSize }: EmptyStateProps) {
  const families = FAMILY_ORDER.map((family: Family) => ({
    family,
    members: methods.filter((m) => m.family === family),
  })).filter((f) => f.members.length > 0);

  return (
    <div className="mx-auto mt-10 max-w-5xl animate-fade-up">
      <p className="mb-5 text-center text-xs uppercase tracking-wider text-ink-faint">
        {methods.length > 0 ? `${methods.length} retrieval methods in ${families.length} families` : "Retrieval methods"}
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {families.map(({ family, members }) => (
          <div key={family} className="rounded-2xl border border-line bg-surface p-5 shadow-soft">
            <h3 className="mb-2 font-display text-base font-semibold text-ink">{FAMILY_LABELS[family]}</h3>
            <p className="mb-3 text-[13px] leading-relaxed text-ink-muted">{FAMILY_BLURBS[family]}</p>
            <div className="flex flex-wrap gap-1.5">
              {members.map((m) => {
                const color = methodColor(m);
                return (
                  <span
                    key={m.key}
                    className="flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] text-ink"
                    style={{ backgroundColor: tint(color, 0x1a) }}
                    title={m.description}
                  >
                    <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
                    {m.label}
                  </span>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-6 text-center text-sm text-ink-faint">
        Type a query above to see how each method ranks the same {corpusSize ?? 727} papers, differently. Pick a part of
        the abstract (task, problem, method or result) to search only that part.
      </p>
    </div>
  );
}
