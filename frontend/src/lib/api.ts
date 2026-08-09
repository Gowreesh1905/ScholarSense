export interface SearchHit {
  rank: number;
  score: number;
  arxiv_id: string;
  url: string;
  snippet: string;
  abstract: string;
}

export type MethodKey = "tfidf" | "word2vec" | "bert";

export interface MethodResult {
  key: MethodKey;
  label: string;
  description: string;
  results: SearchHit[];
}

export interface SearchResponse {
  query: string;
  took_ms: number;
  methods: MethodResult[];
}

export interface MethodInfo {
  key: MethodKey;
  label: string;
  description: string;
}

export interface HealthResponse {
  status: string;
  corpus_size: number;
  device: string;
  methods: MethodInfo[];
}

const API_BASE = "/api";

async function parseErrorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body?.detail) return String(body.detail);
  } catch {
    // response wasn't JSON — fall through to the generic message
  }
  return `Request failed (${res.status})`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await parseErrorMessage(res));
  return res.json();
}

export async function runSearch(query: string, k = 5): Promise<SearchResponse> {
  const res = await fetch(`${API_BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!res.ok) throw new Error(await parseErrorMessage(res));
  return res.json();
}
