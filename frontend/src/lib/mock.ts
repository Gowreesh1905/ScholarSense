// Offline stand-in for the backend, enabled with `VITE_MOCK=1 npm run dev`.
// Responses follow team/CONTRACT.md §6 and §8 exactly, including papers without
// an arXiv ID and aspect results with matched spans, so every UI state can be seen.

import type {
  ClusterSummary,
  HealthResponse,
  MethodInfo,
  ResultsSummary,
  ScaleSummary,
  SearchHit,
  SearchOptions,
  SearchResponse,
} from "./api";

const METHODS: MethodInfo[] = [
  { key: "tfidf", label: "TF-IDF", family: "lexical", score_type: "cosine", color: "#d97706", description: "Keyword matching on lemmatised words, weighted by how rare each word is in the corpus." },
  { key: "bm25", label: "BM25", family: "lexical", score_type: "bm25", color: "#dc2626", description: "Keyword matching with term-frequency saturation and length normalisation." },
  { key: "word2vec", label: "Word2Vec (corpus-trained)", family: "static", score_type: "cosine", color: "#0891b2", description: "Average of word vectors learned from only these 727 abstracts." },
  { key: "glove", label: "GloVe 300d (pretrained)", family: "static", score_type: "cosine", color: "#059669", description: "Average of pretrained GloVe 300d word vectors." },
  { key: "specter", label: "SPECTER", family: "contextual", score_type: "cosine", color: "#9333ea", description: "Transformer trained on citation links between scientific papers." },
  { key: "bge", label: "BGE-base", family: "contextual", score_type: "cosine", color: "#2563eb", description: "Dense retrieval model (BAAI/bge-base-en-v1.5)." },
  { key: "hybrid_rrf", label: "BM25 + BGE (RRF)", family: "hybrid", score_type: "rrf", color: "#64748b", description: "Reciprocal Rank Fusion over BM25 and BGE." },
  { key: "hybrid", label: "Hybrid + re-rank", family: "hybrid", score_type: "cross-encoder", color: "#db2777", description: "BM25 + BGE fused with RRF, re-ranked with a cross-encoder." },
];

const PAPERS: { arxiv_id: string | null; abstract: string; span: [number, number] }[] = [
  {
    arxiv_id: "2111.12503",
    abstract:
      "We present an efficient method for joint optimization of topology, materials and lighting from multi-view image observations. Unlike recent multi-view reconstruction approaches, which typically produce entangled 3D representations encoded in neural networks, we output triangle meshes with spatially-varying materials and environment lighting that can be deployed in any traditional graphics engine unmodified.",
    span: [127, 270],
  },
  {
    arxiv_id: null,
    abstract:
      "Motion blur is a common problem when photographing moving objects with hand-held cameras. Existing deblurring networks struggle with large, non-uniform blur. We propose a recurrent architecture that progressively restores sharp frames and outperforms prior work on standard benchmarks.",
    span: [91, 156],
  },
  {
    arxiv_id: "2112.05957",
    abstract:
      "Robotic grasping of novel objects remains challenging because grasp detectors overfit to the training objects. We learn an object-agnostic grasp quality function from simulated depth images and transfer it to a real robot, achieving a 92% success rate on unseen household items.",
    span: [0, 110],
  },
  {
    arxiv_id: "2203.01234",
    abstract:
      "LiDAR point clouds are sparse at long range, which makes 3D object detection for autonomous driving unreliable for distant vehicles. We densify distant regions with a learned completion module and report consistent gains on the KITTI and Waymo benchmarks.",
    span: [0, 120],
  },
  {
    arxiv_id: "2101.06860",
    abstract:
      "Self-supervised representation learning removes the need for human labels. We show that a simple masked-image objective learns features that transfer to detection and segmentation as well as supervised pretraining does.",
    span: [0, 71],
  },
  {
    arxiv_id: "2111.13260",
    abstract:
      "Reconstructing a 3D model from a single photograph is ill-posed. We combine a learned shape prior with differentiable rendering to recover textured meshes from one image, without any 3D supervision.",
    span: [0, 61],
  },
];

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function seededScores(seed: string, n: number): number[] {
  let h = 2166136261;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return Array.from({ length: n }, () => {
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h >>> 0) % 1000) / 1000;
  });
}

function scoreFor(info: MethodInfo, base: number): number {
  switch (info.score_type) {
    case "bm25":
      return 4 + base * 14;
    case "rrf":
      return 0.016 + base * 0.017;
    case "cross-encoder":
      return -6 + base * 13;
    default:
      return 0.2 + base * 0.7;
  }
}

function hits(seed: string, k: number, info: MethodInfo, withSpan: boolean, withTitle = false): SearchHit[] {
  const scores = seededScores(seed, PAPERS.length);
  const order = PAPERS.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, k).map((doc_index, i) => {
    const p = PAPERS[doc_index];
    return {
      rank: i + 1,
      doc_index,
      score: scoreFor(info, scores[doc_index]),
      arxiv_id: p.arxiv_id,
      url: p.arxiv_id ? `https://arxiv.org/abs/${p.arxiv_id}` : null,
      snippet: p.abstract.length > 300 ? `${p.abstract.slice(0, 300)}…` : p.abstract,
      abstract: p.abstract,
      title: withTitle && doc_index % 2 === 0 ? `A mock distractor paper title #${doc_index}` : null,
      matched_span: withSpan
        ? { start: p.span[0], end: p.span[1], text: p.abstract.slice(p.span[0], p.span[1]) }
        : null,
    };
  });
}

export async function mockFetchHealth(): Promise<HealthResponse> {
  await delay(250);
  return {
    status: "ready",
    corpus_size: 727,
    device: "cuda",
    methods: METHODS,
    default_methods: ["tfidf", "bm25", "bge", "hybrid"],
    aspects: ["all", "task", "problem", "method", "result"],
    aspect_coverage: { task: 465, problem: 452, method: 640, result: 610 },
    corpora: [
      { key: "727", label: "727 papers", size: 727, methods: METHODS.map((m) => m.key), default_methods: ["tfidf", "bm25", "bge", "hybrid"], aspects: true },
      { key: "scale", label: "100,727 papers", size: 100727, methods: ["bm25", "specter", "bge", "hybrid_rrf", "hybrid"], default_methods: ["bm25", "bge", "hybrid"], aspects: false },
    ],
  };
}

export async function mockRunSearch(query: string, { k = 5, methods, aspect = "all", corpus = "727" }: SearchOptions): Promise<SearchResponse> {
  await delay(400);
  const scale = corpus !== "727";
  if (scale && aspect !== "all") throw new Error("Aspect search only works on the 727-paper corpus.");
  const keys = methods?.length ? methods : scale ? ["bm25", "bge", "hybrid"] : ["tfidf", "bm25", "bge", "hybrid"];
  const results = keys.map((key) => {
    const info = METHODS.find((m) => m.key === key);
    if (!info) throw new Error(`Unknown or unavailable method(s): ${key}`);
    return { ...info, took_ms: Math.round(Math.random() * 200) / 10, results: hits(query + key + corpus, k, info, false, scale) };
  });
  if (aspect !== "all") {
    const info: MethodInfo = {
      key: "aspect",
      label: `Aspect: ${aspect[0].toUpperCase()}${aspect.slice(1)}`,
      family: "contextual",
      score_type: "cosine",
      color: "#0f766e",
      description: `BGE similarity against only the ${aspect} each paper states.`,
    };
    results.push({ ...info, took_ms: 4.2, results: hits(query + aspect, k, info, true) });
  }
  return { query, aspect, corpus, took_ms: results.reduce((t, m) => t + m.took_ms, 0), methods: results };
}

const MOCK_SCALE: ScaleSummary = {
  generated_at: "2026-10-05T14:00:00",
  corpus_size: 100727,
  n_distractors: 100000,
  precision: "fp16",
  gpu: "Mock GPU",
  query_sets: { exact: 1030, paraphrased: 1030 },
  rows: (["bm25", "specter", "bge", "hybrid_rrf", "hybrid"] as const).flatMap((key) => {
    const m = METHODS.find((x) => x.key === key)!;
    return (["exact", "paraphrased"] as const).map((set, i) => ({
      method: key, label: m.label, color: m.color, query_set: set,
      "mrr@10_727": [0.52, 0.44][i], "mrr@10_scale": [0.3, 0.24][i],
      "recall@10_727": [0.7, 0.62][i], "recall@10_scale": [0.45, 0.38][i],
      latency_ms_727: 3, latency_ms_scale: 12, latency_ms_p95_scale: 20,
    }));
  }),
  charts: [{ file: "scale/mrr_727_vs_scale.png", title: "MRR@10 at 727 vs. scale", caption: "Mock caption." }],
};

const MOCK_CLUSTER: ClusterSummary = {
  generated_at: "2026-10-05T14:00:00",
  scaling: {
    laptops: ["laptopA", "laptopB", "laptopC"], limit: 30000, precision: "fp16", warmup_s: 22,
    rows: [
      { laptops: ["laptopA"], n: 1, seconds: 260, docs_per_s: 115, speedup: 1, efficiency: 1, ideal_speedup: 1, cluster_efficiency: 1 },
      { laptops: ["laptopB"], n: 1, seconds: 300, docs_per_s: 100, speedup: 0.87, efficiency: 0.87, ideal_speedup: 0.87, cluster_efficiency: 1 },
      { laptops: ["laptopC"], n: 1, seconds: 290, docs_per_s: 103, speedup: 0.9, efficiency: 0.9, ideal_speedup: 0.9, cluster_efficiency: 1 },
      { laptops: ["laptopA", "laptopB"], n: 2, seconds: 150, docs_per_s: 200, speedup: 1.73, efficiency: 0.87, serial_fraction: 0.15, ideal_speedup: 1.87, cluster_efficiency: 0.93 },
      { laptops: ["laptopA", "laptopB", "laptopC"], n: 3, seconds: 108, docs_per_s: 278, speedup: 2.41, efficiency: 0.8, serial_fraction: 0.12, ideal_speedup: 2.77, cluster_efficiency: 0.87 },
    ],
  },
  main_run: {
    run: "full", texts: 101454, laptops: ["laptopA", "laptopB", "laptopC"], precision: "fp16", compute_s: 330, warmup_s: 25,
    received_mb: 420, docs_per_s: 307,
    gpus: { "laptopA-gpu": "RTX 4060", "laptopB-gpu": "RTX 4050", "laptopC-gpu": "RTX 4060" },
    per_laptop: {
      laptopA: { embed_chunks: 150, tokenize_chunks: 140, docs_embedded: 37500 },
      laptopB: { embed_chunks: 120, tokenize_chunks: 130, docs_embedded: 30000 },
      laptopC: { embed_chunks: 136, tokenize_chunks: 136, docs_embedded: 33954 },
    },
    events: [],
  },
  fault_run: {
    run: "demo", texts: 31454, laptops: ["laptopA", "laptopB", "laptopC"], precision: "fp16", compute_s: 140, received_mb: 130, docs_per_s: 224,
    per_laptop: {}, events: [{ t: 40, event: "left", worker: "laptopB-gpu", laptop: "laptopB" }],
  },
  precision_check: { fp16_ok: true, tolerance: 0.005, rows: [{ method: "bge", query_set: "exact", difference: 0.0003 }] },
  charts: [{ file: "cluster/scaling.png", title: "Speedup on 1 to n laptops", caption: "Mock caption." }],
};

export function mockResultFile(path: string): unknown {
  if (path === "scale/summary.json") return MOCK_SCALE;
  if (path === "cluster/summary.json") return MOCK_CLUSTER;
  return null;
}

export async function mockFetchResults(): Promise<ResultsSummary> {
  await delay(250);
  const mrr: Record<string, [number, number]> = {
    tfidf: [0.519, 0.438], bm25: [0.523, 0.439], word2vec: [0.272, 0.215], glove: [0.237, 0.208],
    specter: [0.233, 0.197], bge: [0.498, 0.462], hybrid_rrf: [0.56, 0.507], hybrid: [0.548, 0.506],
  };
  const latency: Record<string, number> = {
    tfidf: 2.1, bm25: 2.5, word2vec: 0.9, glove: 0.6, specter: 9.4, bge: 10.6, hybrid_rrf: 17.2, hybrid: 150.5,
  };
  return {
    generated_at: "2026-10-04T14:38:34",
    device: "cuda",
    gpu: "Mock GPU",
    corpus_size: 727,
    query_sets: {
      exact: { n: 1030, corpus: "masked", description: "Task/problem spans searched in the masked corpus." },
      paraphrased: { n: 1030, corpus: "masked", description: "The same spans back-translated English-German-English." },
    },
    rows: METHODS.flatMap((m) =>
      (["exact", "paraphrased"] as const).map((set, i) => ({
        method: m.key,
        label: m.label,
        family: m.family,
        color: m.color,
        query_set: set,
        "mrr@10": mrr[m.key][i],
        "recall@10": Math.min(1, mrr[m.key][i] + 0.2),
        latency_ms_mean: latency[m.key],
        latency_ms_p95: latency[m.key] * 1.6,
      })),
    ),
    ablation: ["bge", "hybrid_rrf", "hybrid"],
    charts: [
      { file: "overlap_vs_mrr.png", title: "Retrieval quality vs. word overlap", caption: "Mock caption." },
      { file: "speed_vs_accuracy.png", title: "Speed vs. accuracy", caption: "Mock caption." },
    ],
    examples: [
      {
        qid: "para-0143",
        query: "Embeddings that reconstruct a picture well are not always robust in machining processes.",
        query_set: "paraphrased",
        relevant_docs: [99],
        ranks: { tfidf: 64, bm25: 81, word2vec: null, glove: null, specter: 5, bge: 1, hybrid_rrf: 5, hybrid: 1 },
        explanation: "Mock explanation: the query shares one content word with the paper, so keyword methods miss it.",
      },
    ],
  };
}
