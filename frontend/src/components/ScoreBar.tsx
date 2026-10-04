import type { ScoreType } from "../lib/api";

interface ScoreBarProps {
  score: number;
  scoreType: ScoreType;
  color: string;
  /** Lowest and highest score in this column, for scores that aren't on a 0–1 scale. */
  range: [number, number];
}

const SCORE_TYPE_HINT: Record<ScoreType, string> = {
  cosine: "Cosine similarity (0–1)",
  bm25: "BM25 score (unbounded, relative to the top result)",
  rrf: "Reciprocal Rank Fusion score (relative to the top result)",
  "cross-encoder": "Cross-encoder relevance logit (relative to this column)",
};

function barFraction(score: number, scoreType: ScoreType, [min, max]: [number, number]): number {
  if (scoreType === "cosine") return score;
  if (min < 0) {
    // Logits can be negative: scale between the column's lowest and highest score.
    return max === min ? 1 : 0.08 + (0.92 * (score - min)) / (max - min);
  }
  return max > 0 ? score / max : 0;
}

function formatScore(score: number, scoreType: ScoreType): string {
  if (scoreType === "rrf") return score.toFixed(4);
  if (scoreType === "cosine") return score.toFixed(3);
  return score.toFixed(2);
}

export function ScoreBar({ score, scoreType, color, range }: ScoreBarProps) {
  const pct = Math.max(0, Math.min(1, barFraction(score, scoreType, range))) * 100;

  return (
    <div className="flex items-center gap-2" title={SCORE_TYPE_HINT[scoreType]}>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-14 shrink-0 text-right font-mono text-[11px] tabular-nums" style={{ color }}>
        {formatScore(score, scoreType)}
      </span>
    </div>
  );
}
