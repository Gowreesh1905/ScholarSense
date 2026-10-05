import { useEffect, useState } from "react";
import { ChartGrid, Section } from "../components/ResultSection";
import { fetchResults, NoResultsError } from "../lib/api";
import { ClusterSection, ScaleSection } from "./ClusterResults";
import type { ExampleQuery, MethodKey, ResultRow, ResultsSummary } from "../lib/api";
import { FAMILY_LABELS, methodColor, tint } from "../lib/methodStyles";

type LoadState =
  | { kind: "loading" }
  | { kind: "missing" }
  | { kind: "error"; message: string }
  | { kind: "ready"; summary: ResultsSummary };

const SET_LABELS: Record<string, string> = {
  exact: "Exact",
  paraphrased: "Paraphrased",
  handwritten: "Hand-written",
};

const setLabel = (s: string) => SET_LABELS[s] ?? s;
const fmt = (v: number | null | undefined, digits = 3) => (v === null || v === undefined ? "–" : v.toFixed(digits));
const fmtMs = (v: number | null | undefined) => (v === null || v === undefined ? "–" : v < 10 ? v.toFixed(1) : Math.round(v).toString());

interface MethodSummary {
  key: MethodKey;
  label: string;
  family: ResultRow["family"];
  color: string;
  bySet: Record<string, ResultRow>;
  latencyMean: number | null;
  latencyP95: number | null;
}

function groupRows(rows: ResultRow[]): MethodSummary[] {
  const byMethod = new Map<MethodKey, MethodSummary>();
  for (const r of rows) {
    let m = byMethod.get(r.method);
    if (!m) {
      m = { key: r.method, label: r.label, family: r.family, color: methodColor({ key: r.method, color: r.color }), bySet: {}, latencyMean: r.latency_ms_mean, latencyP95: r.latency_ms_p95 };
      byMethod.set(r.method, m);
    }
    m.bySet[r.query_set] = r;
  }
  return [...byMethod.values()];
}

function MethodName({ m }: { m: Pick<MethodSummary, "label" | "color"> }) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: m.color }} />
      <span className="font-medium text-ink">{m.label}</span>
    </span>
  );
}

function ResultsTable({ methods, sets }: { methods: MethodSummary[]; sets: string[] }) {
  const metrics = sets.flatMap((s) => [
    { set: s, field: "mrr@10" as const, header: "MRR@10" },
    { set: s, field: "recall@10" as const, header: "R@10" },
  ]);
  const best = metrics.map(({ set, field }) =>
    Math.max(...methods.map((m) => m.bySet[set]?.[field] ?? -Infinity)),
  );
  const fastest = Math.min(...methods.map((m) => m.latencyMean ?? Infinity));

  return (
    <div className="overflow-x-auto rounded-2xl border border-line bg-surface shadow-soft">
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-[11px] uppercase tracking-wider text-ink-faint">
            <th rowSpan={2} className="px-4 py-2.5 text-left font-medium">Method</th>
            {sets.map((s) => (
              <th key={s} colSpan={2} className="border-l border-line px-3 pt-2.5 pb-1 text-center font-medium">
                {setLabel(s)}
              </th>
            ))}
            <th rowSpan={2} className="border-l border-line px-3 py-2.5 text-right font-medium">
              Latency<br />
              <span className="normal-case tracking-normal">mean ms</span>
            </th>
          </tr>
          <tr className="border-b border-line text-[11px] text-ink-faint">
            {metrics.map((c, i) => (
              <th key={`${c.set}-${c.field}`} className={`px-3 pb-2 text-right font-medium ${i % 2 === 0 ? "border-l border-line" : ""}`}>
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {methods.map((m) => (
            <tr key={m.key} className="border-b border-line last:border-0 hover:bg-surface-hover">
              <td className="px-4 py-2.5">
                <MethodName m={m} />
                <span className="ml-4 text-[11px] text-ink-faint">{FAMILY_LABELS[m.family] ?? m.family}</span>
              </td>
              {metrics.map(({ set, field }, i) => {
                const v = m.bySet[set]?.[field];
                const isBest = v !== undefined && v === best[i];
                return (
                  <td
                    key={`${set}-${field}`}
                    className={`px-3 py-2.5 text-right font-mono tabular-nums ${i % 2 === 0 ? "border-l border-line" : ""} ${isBest ? "font-semibold text-ink" : "text-ink-muted"}`}
                  >
                    {fmt(v)}
                  </td>
                );
              })}
              <td
                className={`border-l border-line px-3 py-2.5 text-right font-mono tabular-nums ${m.latencyMean === fastest ? "font-semibold text-ink" : "text-ink-muted"}`}
                title={m.latencyP95 !== null ? `p95 ${fmtMs(m.latencyP95)} ms` : undefined}
              >
                {fmtMs(m.latencyMean)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AblationTable({ stages, sets }: { stages: MethodSummary[]; sets: string[] }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-line bg-surface shadow-soft">
      <table className="w-full min-w-[480px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-line text-[11px] uppercase tracking-wider text-ink-faint">
            <th className="px-4 py-2.5 text-left font-medium">Stage</th>
            {sets.map((s) => (
              <th key={s} className="border-l border-line px-3 py-2.5 text-right font-medium">
                {setLabel(s)} MRR@10
              </th>
            ))}
            <th className="border-l border-line px-3 py-2.5 text-right font-medium">Latency ms</th>
          </tr>
        </thead>
        <tbody>
          {stages.map((m, i) => (
            <tr key={m.key} className="border-b border-line last:border-0">
              <td className="px-4 py-2.5">
                <span className="flex items-center gap-2">
                  {i > 0 && <span className="text-ink-faint">+</span>}
                  <MethodName m={m} />
                </span>
              </td>
              {sets.map((s) => {
                const v = m.bySet[s]?.["mrr@10"];
                const prev = i > 0 ? stages[i - 1].bySet[s]?.["mrr@10"] : undefined;
                const delta = v !== undefined && prev !== undefined ? v - prev : null;
                return (
                  <td key={s} className="border-l border-line px-3 py-2.5 text-right font-mono tabular-nums text-ink-muted">
                    {fmt(v)}
                    {delta !== null && (
                      <span className={`ml-2 text-[11px] ${delta > 0.0005 ? "text-emerald-600 dark:text-emerald-400" : delta < -0.0005 ? "text-red-500" : "text-ink-faint"}`}>
                        {delta >= 0 ? "+" : "−"}
                        {Math.abs(delta).toFixed(3)}
                      </span>
                    )}
                  </td>
                );
              })}
              <td className="border-l border-line px-3 py-2.5 text-right font-mono tabular-nums text-ink-muted">{fmtMs(m.latencyMean)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExampleCard({ example, methods }: { example: ExampleQuery; methods: MethodSummary[] }) {
  return (
    <article className="rounded-2xl border border-line bg-surface p-5 shadow-soft">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-ink-faint">
        <span className="rounded-full border border-line px-2 py-0.5">{setLabel(example.query_set)} query</span>
        <span className="font-mono">{example.qid}</span>
      </div>
      <p className="mb-3 font-display text-[15px] leading-snug text-ink">&ldquo;{example.query}&rdquo;</p>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {methods
          .filter((m) => m.key in example.ranks)
          .map((m) => {
            const rank = example.ranks[m.key];
            const hit = rank !== null && rank <= 10;
            return (
              <span
                key={m.key}
                className="flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px]"
                style={{ borderColor: hit ? tint(m.color, 0x80) : "var(--line)", backgroundColor: hit ? tint(m.color, 0x14) : "transparent" }}
                title={rank === null ? "Not in the top 100" : `Correct paper ranked #${rank}`}
              >
                <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: m.color }} />
                <span className="text-ink-muted">{m.label}</span>
                <span className={`font-mono tabular-nums ${hit ? "font-semibold text-ink" : "text-ink-faint"}`}>
                  {rank === null ? "–" : `#${rank}`}
                </span>
              </span>
            );
          })}
      </div>
      <p className="text-[13px] leading-relaxed text-ink-muted">{example.explanation}</p>
    </article>
  );
}

export function ResultsPage() {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    fetchResults()
      .then((summary) => !cancelled && setState({ kind: "ready", summary }))
      .catch((err) => {
        if (cancelled) return;
        if (err instanceof NoResultsError) setState({ kind: "missing" });
        else setState({ kind: "error", message: err instanceof Error ? err.message : "Could not load results." });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.kind === "loading") {
    return (
      <div className="flex flex-col gap-4">
        <div className="skeleton h-8 w-64 rounded" />
        <div className="skeleton h-64 w-full rounded-2xl" />
      </div>
    );
  }

  if (state.kind === "missing" || state.kind === "error") {
    return (
      <div className="mx-auto mt-10 max-w-xl rounded-2xl border border-dashed border-line p-8 text-center">
        <p className="font-display text-lg font-semibold text-ink">
          {state.kind === "missing" ? "Results not generated yet" : "Could not load results"}
        </p>
        <p className="mt-2 text-sm text-ink-muted">
          {state.kind === "missing" ? (
            <>
              Run <code className="rounded bg-surface-hover px-1 py-0.5 font-mono text-[13px]">python eval/run_eval.py</code> and{" "}
              <code className="rounded bg-surface-hover px-1 py-0.5 font-mono text-[13px]">python eval/charts.py</code>, then reload this page.
            </>
          ) : (
            <>
              {state.message}. Is the API running? Try{" "}
              <code className="rounded bg-surface-hover px-1 py-0.5 font-mono text-[13px]">python backend/app.py</code>.
            </>
          )}
        </p>
      </div>
    );
  }

  const { summary } = state;
  const sets = Object.keys(summary.query_sets);
  const methods = groupRows(summary.rows);
  const stages = summary.ablation
    .map((key) => methods.find((m) => m.key === key))
    .filter((m): m is MethodSummary => m !== undefined);
  const generated = new Date(summary.generated_at);

  return (
    <div className="animate-fade-up">
      <div className="mb-10">
        <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">Evaluation results</h1>
        <p className="mt-2 max-w-3xl text-[15px] text-ink-muted">
          How often each method puts the right paper near the top. <strong className="font-medium text-ink">MRR@10</strong> rewards
          ranking it first (1 = always #1); <strong className="font-medium text-ink">R@10</strong> is how often it is in the top 10.
        </p>
        <div className="mt-4 flex flex-wrap gap-2 text-xs text-ink-muted">
          <span className="rounded-full border border-line bg-surface px-3 py-1">{summary.corpus_size} papers</span>
          {sets.map((s) => (
            <span key={s} className="rounded-full border border-line bg-surface px-3 py-1" title={summary.query_sets[s].description}>
              {setLabel(s)}: {summary.query_sets[s].n.toLocaleString()} queries ({summary.query_sets[s].corpus} corpus)
            </span>
          ))}
          <span className="rounded-full border border-line bg-surface px-3 py-1">
            Latency on {summary.gpu ?? summary.device.toUpperCase()}
          </span>
          {!Number.isNaN(generated.getTime()) && (
            <span className="rounded-full border border-line bg-surface px-3 py-1">Generated {generated.toLocaleString()}</span>
          )}
        </div>
        <ul className="mt-4 max-w-3xl list-disc space-y-1 pl-5 text-[13px] text-ink-muted">
          {sets.map((s) => (
            <li key={s}>
              <span className="font-medium text-ink">{setLabel(s)}:</span> {summary.query_sets[s].description}
            </li>
          ))}
        </ul>
      </div>

      <Section title="All methods" subtitle="Bold = best in the column (highest accuracy, lowest latency). Hover a latency to see its 95th percentile.">
        <ResultsTable methods={methods} sets={sets} />
      </Section>

      {stages.length > 1 && (
        <Section
          title="Hybrid ablation"
          subtitle="What each stage of the hybrid pipeline adds: dense retrieval alone, then fused with BM25 keyword ranking (Reciprocal Rank Fusion), then re-ranked by a cross-encoder. Changes are relative to the stage above."
        >
          <AblationTable stages={stages} sets={sets} />
        </Section>
      )}

      {summary.charts.length > 0 && (
        <Section title="Charts">
          <ChartGrid charts={summary.charts} />
        </Section>
      )}

      {summary.examples.length > 0 && (
        <Section title="Where methods win and fail" subtitle="Rank of the correct paper for each method (– = not in the top 100). Highlighted = in the top 10.">
          <div className="grid grid-cols-1 gap-4">
            {summary.examples.map((e) => (
              <ExampleCard key={e.qid} example={e} methods={methods} />
            ))}
          </div>
        </Section>
      )}

      <ScaleSection />
      <ClusterSection />
    </div>
  );
}
