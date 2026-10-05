import { useState } from "react";
import type { MatchedSpan, ScoreType, SearchHit } from "../lib/api";
import { tint } from "../lib/methodStyles";
import { ScoreBar } from "./ScoreBar";

interface ResultCardProps {
  hit: SearchHit;
  color: string;
  scoreType: ScoreType;
  scoreRange: [number, number];
  foundByOtherMethods: string[];
  index: number;
}

const CONTEXT_CHARS = 75;

/** Move `i` outward to the nearest space so the excerpt doesn't start or end mid-word. */
function snapToSpace(text: string, i: number, direction: -1 | 1): number {
  if (i <= 0) return 0;
  if (i >= text.length) return text.length;
  const found = direction < 0 ? text.lastIndexOf(" ", i) : text.indexOf(" ", i);
  return found === -1 ? (direction < 0 ? 0 : text.length) : found;
}

function Highlighted({ text, span, color }: { text: string; span: MatchedSpan; color: string }) {
  return (
    <>
      {text.slice(0, span.start)}
      <mark className="rounded px-0.5 text-ink" style={{ backgroundColor: tint(color, 0x38) }}>
        {text.slice(span.start, span.end)}
      </mark>
      {text.slice(span.end)}
    </>
  );
}

/** Collapsed view of a matched span: the span plus a little context on each side. */
function SpanExcerpt({ abstract, span, color }: { abstract: string; span: MatchedSpan; color: string }) {
  const from = snapToSpace(abstract, span.start - CONTEXT_CHARS, -1);
  const to = snapToSpace(abstract, span.end + CONTEXT_CHARS, 1);
  const excerpt = abstract.slice(from, to).trim();
  const offset = abstract.slice(from, to).indexOf(excerpt) + from;
  return (
    <>
      {from > 0 && "… "}
      <Highlighted text={excerpt} span={{ ...span, start: span.start - offset, end: span.end - offset }} color={color} />
      {to < abstract.length && " …"}
    </>
  );
}

export function ResultCard({ hit, color, scoreType, scoreRange, foundByOtherMethods, index }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false);
  const span = hit.matched_span;
  const collapsedLength = span ? Math.min(hit.abstract.length, span.end - span.start + 2 * CONTEXT_CHARS) : hit.snippet.length;
  const isTruncated = hit.abstract.length > collapsedLength;

  return (
    <article
      className="animate-fade-up rounded-xl border border-line bg-surface p-4 shadow-soft transition hover:border-line-strong"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="mb-2 flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold"
            style={{ backgroundColor: tint(color, 0x1f), color }}
          >
            {hit.rank}
          </span>
          {hit.url ? (
            <a
              href={hit.url}
              target="_blank"
              rel="noreferrer"
              className="truncate font-mono text-xs text-ink-muted transition hover:text-accent hover:underline"
            >
              arXiv:{hit.arxiv_id}
            </a>
          ) : (
            <span className="truncate text-xs italic text-ink-faint" title={`Paper #${hit.doc_index} has no arXiv ID in the dataset`}>
              No arXiv ID
            </span>
          )}
        </div>
        {hit.url && (
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
        )}
      </div>

      <p className="mb-2.5 text-[13px] leading-relaxed text-ink-muted">
        {span ? (
          expanded ? (
            <Highlighted text={hit.abstract} span={span} color={color} />
          ) : (
            <SpanExcerpt abstract={hit.abstract} span={span} color={color} />
          )
        ) : expanded ? (
          hit.abstract
        ) : (
          hit.snippet
        )}
        {isTruncated && (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="ml-1.5 font-medium hover:underline"
            style={{ color }}
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

      <ScoreBar score={hit.score} scoreType={scoreType} color={color} range={scoreRange} />
    </article>
  );
}
