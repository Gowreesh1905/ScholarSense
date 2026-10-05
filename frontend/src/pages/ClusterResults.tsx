import { useEffect, useState } from "react";
import { ChartGrid, Pill, Section } from "../components/ResultSection";
import { fetchResultFile } from "../lib/api";
import type { ClusterRun, ClusterSummary, ScaleRow, ScaleSummary } from "../lib/api";

const SET_LABELS: Record<string, string> = { exact: "Exact", paraphrased: "Paraphrased", handwritten: "Hand-written" };
const fmt3 = (v: number | null | undefined) => (v === null || v === undefined ? "–" : v.toFixed(3));
const pct = (v: number | undefined) => (v === undefined ? "–" : `${Math.round(v * 100)}%`);
const secs = (v: number) => (v < 100 ? `${v.toFixed(1)} s` : `${Math.round(v)} s`);

/** Loads an optional results file; undefined while loading, null if it doesn't exist (yet). */
function useResultFile<T>(path: string): T | null | undefined {
  const [data, setData] = useState<T | null | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    fetchResultFile<T>(path)
      .then((d) => !cancelled && setData(d))
      .catch(() => !cancelled && setData(null));
    return () => {
      cancelled = true;
    };
  }, [path]);
  return data;
}

function Table({ head, children, minWidth = 560 }: { head: React.ReactNode; children: React.ReactNode; minWidth?: number }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-line bg-surface shadow-soft">
      <table className="w-full border-collapse text-sm" style={{ minWidth }}>
        <thead>
          <tr className="border-b border-line text-[11px] uppercase tracking-wider text-ink-faint">{head}</tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

const th = "px-3 py-2.5 text-right font-medium";
const td = "px-3 py-2.5 text-right font-mono tabular-nums text-ink-muted";

// ---------------------------------------------------------------------------
// Evaluation at scale
// ---------------------------------------------------------------------------

function Change({ before, after }: { before: number | null; after: number }) {
  if (before === null) return null;
  const d = after - before;
  return (
    <span className={`ml-1.5 text-[11px] ${d < -0.0005 ? "text-red-500" : d > 0.0005 ? "text-emerald-600 dark:text-emerald-400" : "text-ink-faint"}`}>
      {d >= 0 ? "+" : "−"}
      {Math.abs(d).toFixed(3)}
    </span>
  );
}

export function ScaleSection() {
  const summary = useResultFile<ScaleSummary>("scale/summary.json");
  if (!summary) return null;

  const sets = Object.keys(summary.query_sets);
  const methods = [...new Map(summary.rows.map((r) => [r.method, r])).values()];
  const row = (m: string, s: string): ScaleRow | undefined => summary.rows.find((r) => r.method === m && r.query_set === s);

  return (
    <Section
      title={`At ${summary.corpus_size.toLocaleString()} papers`}
      subtitle={
        <>
          The same queries and the same correct papers, now hidden among {summary.n_distractors.toLocaleString()} arXiv
          computer-science abstracts from 2021–2022 that the laptop cluster embedded. Every extra paper is another
          candidate that can outrank the right one, so scores drop; the question is which methods hold up.
        </>
      }
    >
      <div className="mb-4 flex flex-wrap gap-2 text-xs text-ink-muted">
        <Pill>{summary.corpus_size.toLocaleString()} papers</Pill>
        {sets.map((s) => (
          <Pill key={s}>
            {SET_LABELS[s] ?? s}: {summary.query_sets[s].toLocaleString()} queries
          </Pill>
        ))}
        <Pill title="Document embeddings computed in half precision; see the precision check below">
          Index built in {summary.precision}
        </Pill>
        {summary.gpu && <Pill>Latency on {summary.gpu}</Pill>}
      </div>
      <Table
        minWidth={640}
        head={
          <>
            <th className="px-4 py-2.5 text-left font-medium">Method</th>
            {sets.map((s) => (
              <th key={s} className={`${th} border-l border-line`}>
                {SET_LABELS[s] ?? s} MRR@10
                <br />
                <span className="normal-case tracking-normal">727 → {summary.corpus_size.toLocaleString()}</span>
              </th>
            ))}
            {sets.map((s) => (
              <th key={`r-${s}`} className={`${th} border-l border-line`}>
                {SET_LABELS[s] ?? s} R@10
              </th>
            ))}
            <th className={`${th} border-l border-line`}>Latency ms</th>
          </>
        }
      >
        {methods.map((m) => (
          <tr key={m.method} className="border-b border-line last:border-0 hover:bg-surface-hover">
            <td className="px-4 py-2.5">
              <span className="flex items-center gap-2">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: m.color }} />
                <span className="font-medium text-ink">{m.label}</span>
              </span>
            </td>
            {sets.map((s) => {
              const r = row(m.method, s);
              return (
                <td key={s} className={`${td} border-l border-line`}>
                  {r ? (
                    <>
                      <span className="text-ink-faint">{fmt3(r["mrr@10_727"])} →</span> <span className="text-ink">{fmt3(r["mrr@10_scale"])}</span>
                      <Change before={r["mrr@10_727"]} after={r["mrr@10_scale"]} />
                    </>
                  ) : (
                    "–"
                  )}
                </td>
              );
            })}
            {sets.map((s) => {
              const r = row(m.method, s);
              return (
                <td key={`r-${s}`} className={`${td} border-l border-line`}>
                  {r ? `${fmt3(r["recall@10_727"])} → ${fmt3(r["recall@10_scale"])}` : "–"}
                </td>
              );
            })}
            <td className={`${td} border-l border-line`} title={`p95 ${m.latency_ms_p95_scale} ms`}>
              {m.latency_ms_scale.toFixed(1)}
            </td>
          </tr>
        ))}
      </Table>
      {summary.charts.length > 0 && (
        <div className="mt-6">
          <ChartGrid charts={summary.charts} />
        </div>
      )}
    </Section>
  );
}

// ---------------------------------------------------------------------------
// The laptop cluster
// ---------------------------------------------------------------------------

function RunCard({ title, run, children }: { title: string; run: ClusterRun; children?: React.ReactNode }) {
  const laptops = Object.keys(run.per_laptop).sort();
  const totalEmbed = laptops.reduce((t, l) => t + run.per_laptop[l].embed_chunks, 0) || 1;
  const gpuOf = (laptop: string) => run.gpus?.[`${laptop}-gpu`] ?? run.workers?.[laptop]?.gpu.map((g) => run.gpus?.[g]).find(Boolean);

  return (
    <div className="rounded-2xl border border-line bg-surface p-5 shadow-soft">
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="mt-1 text-[13px] text-ink-muted">
        {run.texts.toLocaleString()} texts in <span className="font-medium text-ink">{secs(run.compute_s)}</span>
        {run.warmup_s !== undefined && <> (+{secs(run.warmup_s)} loading models)</>} · {Math.round(run.docs_per_s)} texts/s ·{" "}
        {run.received_mb} MB sent back to laptop A · {run.precision}
      </p>
      {children}
      {laptops.length > 0 && (
        <table className="mt-3 w-full text-[13px]">
          <thead>
            <tr className="text-[11px] uppercase tracking-wider text-ink-faint">
              <th className="py-1 text-left font-medium">Laptop</th>
              <th className="py-1 text-left font-medium">GPU</th>
              <th className="py-1 text-right font-medium">Chunks embedded</th>
              <th className="py-1 text-right font-medium">Chunks tokenized</th>
            </tr>
          </thead>
          <tbody>
            {laptops.map((l) => (
              <tr key={l} className="border-t border-line">
                <td className="py-1.5 font-medium text-ink">{l}</td>
                <td className="py-1.5 text-ink-muted">{gpuOf(l) ?? "–"}</td>
                <td className="py-1.5 text-right font-mono tabular-nums text-ink-muted">
                  {run.per_laptop[l].embed_chunks} <span className="text-ink-faint">({pct(run.per_laptop[l].embed_chunks / totalEmbed)})</span>
                </td>
                <td className="py-1.5 text-right font-mono tabular-nums text-ink-muted">{run.per_laptop[l].tokenize_chunks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function ClusterSection() {
  const summary = useResultFile<ClusterSummary>("cluster/summary.json");
  if (!summary) return null;
  const { scaling, main_run: main, fault_run: fault, precision_check: precision } = summary;

  return (
    <Section
      title="Laptop cluster"
      subtitle={
        <>
          Building the search index means embedding every abstract with BGE and SPECTER (GPU work) and tokenizing it for
          BM25 (CPU work). Dask splits the corpus into chunks and hands each one to whichever laptop is free: GPU chunks to
          a laptop's GPU worker, tokenizing chunks to its CPU workers.
        </>
      }
    >
      {precision && (
        <p className="mb-4 text-[13px] text-ink-muted">
          <span className={precision.fp16_ok ? "font-medium text-emerald-600 dark:text-emerald-400" : "font-medium text-red-500"}>
            {precision.fp16_ok ? "fp16 check passed:" : "fp16 check failed:"}
          </span>{" "}
          half-precision embeddings change MRR@10 by at most{" "}
          {Math.max(...precision.rows.map((r) => Math.abs(r.difference))).toFixed(4)} (tolerance ±{precision.tolerance}).
        </p>
      )}

      {scaling && (
        <div className="mb-6">
          <p className="mb-2 text-sm font-medium text-ink">
            Scaling: the same {scaling.limit.toLocaleString()} abstracts on 1, 2 and 3 laptops
          </p>
          <Table
            minWidth={720}
            head={
              <>
                <th className="px-4 py-2.5 text-left font-medium">Laptops</th>
                <th className={th}>Time</th>
                <th className={th}>Texts/s</th>
                <th className={th}>Speedup</th>
                <th className={th}>Efficiency</th>
                <th className={th} title="Best possible with these GPUs: the laptops' solo speeds added up">Ideal speedup</th>
                <th className={th} title="Achieved time vs. the ideal for these GPUs">Cluster eff.</th>
                <th className={th} title="Karp–Flatt estimate of the share of the job that didn't parallelise">Serial share</th>
              </>
            }
          >
            {scaling.rows.map((r) => (
              <tr key={r.laptops.join("+")} className="border-b border-line last:border-0 hover:bg-surface-hover">
                <td className="px-4 py-2.5 text-ink">{r.laptops.join(" + ")}</td>
                <td className={td}>{secs(r.seconds)}</td>
                <td className={td}>{Math.round(r.docs_per_s)}</td>
                <td className={`${td} font-semibold text-ink`}>{r.speedup !== undefined ? `${r.speedup.toFixed(2)}×` : "–"}</td>
                <td className={td}>{pct(r.efficiency)}</td>
                <td className={td}>{r.ideal_speedup !== undefined ? `${r.ideal_speedup.toFixed(2)}×` : "–"}</td>
                <td className={td}>{pct(r.cluster_efficiency)}</td>
                <td className={td}>{pct(r.serial_fraction)}</td>
              </tr>
            ))}
          </Table>
          <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-ink-faint">
            Speedup = time on {scaling.laptops[0]} alone ÷ time on n laptops. Efficiency = speedup ÷ n. The GPUs aren't
            identical, so "ideal" adds up each laptop's solo speed instead of assuming n×. What stops it reaching ideal is
            the part that can't be split (Amdahl's law): handing out chunks, sending ~4 KB of embeddings per abstract back
            to laptop A over the hotspot, and waiting for the last chunk. Models load once per laptop before timing
            ({secs(scaling.warmup_s)}).
          </p>
        </div>
      )}

      <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {main && <RunCard title={`Index build (run "${main.run}")`} run={main} />}
        {fault && (
          <RunCard title={`Fault tolerance (run "${fault.run}")`} run={fault}>
            <ul className="mt-2 space-y-0.5 text-[13px]">
              {[...new Map(fault.events.filter((e) => e.event === "left").map((e) => [e.laptop, e])).values()].map((e) => (
                <li key={e.laptop} className="text-red-500">
                  {e.laptop} disconnected at {secs(e.t)}; its unfinished chunks were re-run on the other laptops.
                </li>
              ))}
              <li className="text-emerald-600 dark:text-emerald-400">All {fault.texts.toLocaleString()} texts were still processed.</li>
            </ul>
          </RunCard>
        )}
      </div>

      {summary.charts.length > 0 && <ChartGrid charts={summary.charts} />}
    </Section>
  );
}
