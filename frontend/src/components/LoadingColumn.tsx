import { tint } from "../lib/methodStyles";

interface LoadingColumnProps {
  label: string;
  color: string;
  count?: number;
}

export function LoadingColumn({ label, color, count = 5 }: LoadingColumnProps) {
  return (
    <section className="flex min-w-0 flex-1 flex-col">
      <div
        className="mb-3 rounded-xl border px-4 py-3"
        style={{ borderColor: tint(color, 0x40), backgroundColor: tint(color, 0x14) }}
      >
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
          <h2 className="font-display text-[15px] font-semibold leading-snug text-ink">{label}</h2>
        </div>
        <div className="mt-1.5 skeleton h-2.5 w-2/3 rounded" />
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
