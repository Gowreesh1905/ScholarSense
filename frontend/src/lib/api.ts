import { mockFetchHealth, mockFetchResults, mockRunSearch } from "./mock";

// Types mirror team/CONTRACT.md §6 (HTTP API) and §8 (results/summary.json).

export type MethodKey = string;
export type ScoreType = "cosine" | "bm25" | "rrf" | "cross-encoder";
export type Family = "lexical" | "static" | "contextual" | "hybrid";

export interface MatchedSpan {
  start: number;
  end: number;
  text: string;
}

export interface SearchHit {
  rank: number;
  doc_index: number;
  score: number;
  arxiv_id: string | null;
  url: string | null;
  snippet: string;
  abstract: string;
  matched_span: MatchedSpan | null;
}

export interface MethodInfo {
  key: MethodKey;
  label: string;
  family: Family;
  description: string;
  score_type: ScoreType;
  color: string;
}

export interface MethodResult extends MethodInfo {
  took_ms: number;
  results: SearchHit[];
}

export interface SearchResponse {
  query: string;
  aspect: string;
  took_ms: number;
  methods: MethodResult[];
}

export interface HealthResponse {
  status: string;
  corpus_size: number;
  device: string;
  methods: MethodInfo[];
  default_methods: MethodKey[];
  aspects: string[];
  aspect_coverage: Record<string, number>;
}

export interface SearchOptions {
  k?: number;
  methods?: MethodKey[];
  aspect?: string;
}

export interface QuerySetInfo {
  n: number;
  corpus: string;
  description: string;
}

export interface ResultRow {
  method: MethodKey;
  label: string;
  family: Family;
  color: string;
  query_set: string;
  "mrr@10": number;
  "recall@10": number;
  latency_ms_mean: number | null;
  latency_ms_p95: number | null;
}

export interface ChartInfo {
  file: string;
  title: string;
  caption: string;
}

export interface ExampleQuery {
  qid: string;
  query: string;
  query_set: string;
  relevant_docs: number[];
  ranks: Record<MethodKey, number | null>;
  explanation: string;
}

export interface ResultsSummary {
  generated_at: string;
  device: string;
  gpu: string | null;
  corpus_size: number;
  query_sets: Record<string, QuerySetInfo>;
  rows: ResultRow[];
  ablation: MethodKey[];
  charts: ChartInfo[];
  examples: ExampleQuery[];
}

/** Thrown by fetchResults() when the evaluation hasn't been run yet (HTTP 404). */
export class NoResultsError extends Error {}

const API_BASE = "/api";
const USE_MOCK = import.meta.env.VITE_MOCK === "1";

export function resultFileUrl(file: string): string {
  return USE_MOCK ? `data:image/svg+xml,${encodeURIComponent(mockChartSvg(file))}` : `${API_BASE}/results/files/${file}`;
}

async function parseErrorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    // FastAPI validation errors: [{loc, msg, ...}]
    if (Array.isArray(body?.detail) && body.detail[0]?.msg) return String(body.detail[0].msg);
  } catch {
    // response wasn't JSON — fall through to the generic message
  }
  return `Request failed (${res.status})`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  if (USE_MOCK) return mockFetchHealth();
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await parseErrorMessage(res));
  return res.json();
}

export async function runSearch(query: string, { k = 5, methods, aspect = "all" }: SearchOptions = {}): Promise<SearchResponse> {
  if (USE_MOCK) return mockRunSearch(query, { k, methods, aspect });
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k, methods, aspect }),
  });
  if (!res.ok) throw new Error(await parseErrorMessage(res));
  return res.json();
}

export async function fetchResults(): Promise<ResultsSummary> {
  if (USE_MOCK) return mockFetchResults();
  const res = await fetch(`${API_BASE}/results`);
  if (res.status === 404) throw new NoResultsError("No results yet");
  if (!res.ok) throw new Error(await parseErrorMessage(res));
  return res.json();
}

function mockChartSvg(file: string): string {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450"><rect width="100%" height="100%" fill="#fff"/><text x="50%" y="50%" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#888">${file} (mock)</text></svg>`;
}
