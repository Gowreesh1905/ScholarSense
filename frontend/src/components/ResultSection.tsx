import type { ReactNode } from "react";
import { resultFileUrl } from "../lib/api";
import type { ChartInfo } from "../lib/api";

export function Section({ title, subtitle, children }: { title: string; subtitle?: ReactNode; children: ReactNode }) {
  return (
    <section className="mb-12">
      <h2 className="font-display text-xl font-semibold tracking-tight text-ink">{title}</h2>
      {subtitle && <p className="mt-1 mb-4 max-w-3xl text-[13px] leading-relaxed text-ink-muted">{subtitle}</p>}
      {!subtitle && <div className="mb-4" />}
      {children}
    </section>
  );
}

/** Chart PNGs from results/ (white background, so they sit in a white card in both themes). */
export function ChartGrid({ charts }: { charts: ChartInfo[] }) {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {charts.map((c) => (
        <figure key={c.file} className="overflow-hidden rounded-2xl border border-line bg-surface shadow-soft">
          <div className="bg-white p-2">
            <img src={resultFileUrl(c.file)} alt={c.title} loading="lazy" className="h-auto w-full" />
          </div>
          <figcaption className="border-t border-line px-4 py-3">
            <p className="text-sm font-medium text-ink">{c.title}</p>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-muted">{c.caption}</p>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

export function Pill({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span className="rounded-full border border-line bg-surface px-3 py-1" title={title}>
      {children}
    </span>
  );
}
