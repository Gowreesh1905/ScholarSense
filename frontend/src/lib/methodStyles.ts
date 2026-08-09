import type { MethodKey } from "./api";

interface MethodStyle {
  dot: string;
  text: string;
  bgSoft: string;
  border: string;
  bar: string;
  ring: string;
}

export const METHOD_STYLES: Record<MethodKey, MethodStyle> = {
  tfidf: {
    dot: "bg-tfidf",
    text: "text-tfidf",
    bgSoft: "bg-tfidf/10",
    border: "border-tfidf/25",
    bar: "bg-tfidf",
    ring: "focus-visible:ring-tfidf/40",
  },
  word2vec: {
    dot: "bg-word2vec",
    text: "text-word2vec",
    bgSoft: "bg-word2vec/10",
    border: "border-word2vec/25",
    bar: "bg-word2vec",
    ring: "focus-visible:ring-word2vec/40",
  },
  bert: {
    dot: "bg-bert",
    text: "text-bert",
    bgSoft: "bg-bert/10",
    border: "border-bert/25",
    bar: "bg-bert",
    ring: "focus-visible:ring-bert/40",
  },
};

export const METHOD_TAGLINE: Record<MethodKey, string> = {
  tfidf: "Exact word matches only",
  word2vec: "One fixed vector per word",
  bert: "Context-aware, citation-informed",
};
