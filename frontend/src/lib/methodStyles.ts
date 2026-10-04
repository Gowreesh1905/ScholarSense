import type { Family, MethodInfo, MethodKey } from "./api";

// Fallback colors (same as team/CONTRACT.md / searchers/registry.py). The API sends
// `color` with every method; these cover anything rendered before a response arrives.
// Colors are applied with inline styles: Tailwind v4 can't see dynamically built class names.
export const METHOD_COLORS: Record<MethodKey, string> = {
  tfidf: "#d97706",
  bm25: "#dc2626",
  word2vec: "#0891b2",
  glove: "#059669",
  specter: "#9333ea",
  bge: "#2563eb",
  hybrid_rrf: "#64748b",
  hybrid: "#db2777",
  bm25_prf: "#ea580c",
  aspect: "#0f766e",
};

const FALLBACK_COLOR = "#6b7280";

export function methodColor(method: Pick<MethodInfo, "key"> & { color?: string }): string {
  return method.color || METHOD_COLORS[method.key] || FALLBACK_COLOR;
}

/** The color at low opacity, for tinted backgrounds/borders (`alpha` is 0–255). */
export function tint(color: string, alpha: number): string {
  const hex = Math.round(Math.max(0, Math.min(255, alpha)))
    .toString(16)
    .padStart(2, "0");
  return /^#[0-9a-f]{6}$/i.test(color) ? `${color}${hex}` : color;
}

export const FAMILY_ORDER: Family[] = ["lexical", "static", "contextual", "hybrid"];

export const FAMILY_LABELS: Record<Family, string> = {
  lexical: "Lexical",
  static: "Static vectors",
  contextual: "Contextual",
  hybrid: "Hybrid",
};

export const FAMILY_BLURBS: Record<Family, string> = {
  lexical: "Score only words the query and the paper share. \"car\" never matches \"automobile.\"",
  static: "One fixed vector per word; a paper is the average of its words. Knows synonyms, ignores context.",
  contextual: "Transformers that encode the whole sentence, so the same word can mean different things.",
  hybrid: "Combine keyword and meaning-based rankings, then re-read the best candidates with a cross-encoder.",
};
