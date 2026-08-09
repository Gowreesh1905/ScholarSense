import { useState } from "react";
import type { MethodKey, SearchHit } from "../lib/api";
import { METHOD_STYLES } from "../lib/methodStyles";
import { ScoreBar } from "./ScoreBar";

interface ResultCardProps {
  hit: SearchHit;
  method: MethodKey;
  foundByOtherMethods: string[];
  index: number;
}

export function ResultCard({ hit, method, foundByOtherMethods, index }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false);
  const style = METHOD_STYLES[method];
  const isTruncated = hit.abstract.length > hit.snippet.length;

  return (
    <article
      className="animate-fade-up rounded-xl border border-line bg-surface p-4 shadow-soft transition hover:border-line-strong"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold ${style.bgSoft} ${style.text}`}>
            {hit.rank}
          </span>
          <a
            href={hit.url}
            target="_blank"
            rel="noreferrer"
            className="font-mono text-xs text-ink-muted transition hover:text-accent hover:underline"
          >
            arXiv:{hit.arxiv_id}
          </a>
        </div>
        <a
          href={hit.url}
          target="_blank"
          rel="noreferrer"
          aria-label="Open on arXiv"
          className="text-ink-faint transition hover:text-accent"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-3.5 w-3.5">
            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
            <path d="M15 3h6v6M10 14 21 3" />
          </svg>
        </a>
      </div>

      <p className="mb-2.5 text-[13px] leading-relaxed text-ink-muted">
        {expanded ? hit.abstract : hit.snippet}
        {isTruncated && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className={`ml-1.5 font-medium ${style.text} hover:underline`}
          >
            {expanded ? "show less" : "read more"}
          </button>
        )}
      </p>

      {foundByOtherMethods.length > 0 && (
        <div className="mb-2.5 flex flex-wrap gap-1">
          {foundByOtherMethods.map((label) => (
            <span
              key={label}
              className="rounded-full border border-line bg-surface-hover px-2 py-0.5 text-[10px] text-ink-faint"
            >
              also in {label}
            </span>
          ))}
        </div>
      )}

      <ScoreBar score={hit.score} method={method} />
    </article>
  );
}
