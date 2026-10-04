import { useCallback, useMemo, useRef, useState } from "react";
import { AspectSelector } from "../components/AspectSelector";
import { EmptyState } from "../components/EmptyState";
import { ErrorBanner } from "../components/ErrorBanner";
import { ExampleQueries } from "../components/ExampleQueries";
import { LoadingColumn } from "../components/LoadingColumn";
import { MethodColumn } from "../components/MethodColumn";
import { ModelPicker } from "../components/ModelPicker";
import { SearchBar } from "../components/SearchBar";
import { runSearch } from "../lib/api";
import type { HealthResponse, MethodKey, SearchResponse } from "../lib/api";
import { methodColor } from "../lib/methodStyles";

const RESULTS_PER_METHOD = 5;

// Static class strings, one per column count (Tailwind can't see dynamically built classes).
// 5 columns = 4 methods + the aspect column.
const GRID_COLS: Record<number, string> = {
  1: "md:grid-cols-1 md:max-w-xl md:mx-auto",
  2: "md:grid-cols-2",
  3: "md:grid-cols-3",
  4: "md:grid-cols-2 xl:grid-cols-4",
  5: "md:grid-cols-3 xl:grid-cols-5",
};

function buildOverlapIndex(response: SearchResponse) {
  const index = new Map<number, Set<MethodKey>>();
  for (const method of response.methods) {
    for (const hit of method.results) {
      if (!index.has(hit.doc_index)) index.set(hit.doc_index, new Set());
      index.get(hit.doc_index)!.add(method.key);
    }
  }
  return index;
}

interface SearchPageProps {
  health: HealthResponse | null;
}

export function SearchPage({ health }: SearchPageProps) {
  const [query, setQuery] = useState("");
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<MethodKey[] | null>(null);
  const [aspect, setAspect] = useState("all");
  const requestId = useRef(0);

  const methods = useMemo(() => health?.methods ?? [], [health]);
  // Columns always appear in the backend's method order, whatever order they were clicked in;
  // keys the backend doesn't offer are dropped.
  const inMethodOrder = useCallback(
    (keys: MethodKey[]) => methods.filter((m) => keys.includes(m.key)).map((m) => m.key),
    [methods],
  );
  // Until the user picks, follow the backend's defaults.
  const selected = useMemo(() => {
    const keys = inMethodOrder(picked ?? health?.default_methods ?? []);
    return keys.length > 0 ? keys : methods.slice(0, 1).map((m) => m.key);
  }, [picked, health, methods, inMethodOrder]);
  const aspects = health?.aspects ?? ["all"];

  const executeSearch = useCallback(async (q: string, methodKeys: MethodKey[], aspectName: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;

    const thisRequest = ++requestId.current;
    setLoading(true);
    setError(null);
    setLastQuery(trimmed);
    try {
      const response = await runSearch(trimmed, {
        k: RESULTS_PER_METHOD,
        methods: methodKeys.length > 0 ? methodKeys : undefined,
        aspect: aspectName,
      });
      if (thisRequest === requestId.current) setResults(response);
    } catch (err) {
      if (thisRequest === requestId.current) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
        setResults(null);
      }
    } finally {
      if (thisRequest === requestId.current) setLoading(false);
    }
  }, []);

  function handleExamplePick(q: string) {
    setQuery(q);
    executeSearch(q, selected, aspect);
  }

  function handleMethodsChange(next: MethodKey[]) {
    setPicked(next);
    if (lastQuery) executeSearch(lastQuery, inMethodOrder(next), aspect);
  }

  function handleAspectChange(next: string) {
    setAspect(next);
    if (lastQuery) executeSearch(lastQuery, selected, next);
  }

  const overlapIndex = useMemo(() => (results ? buildOverlapIndex(results) : new Map<number, Set<MethodKey>>()), [results]);
  const labelByKey = useMemo(() => new Map(results?.methods.map((m) => [m.key, m.label]) ?? []), [results]);

  function overlapLabelsFor(method: MethodKey) {
    return (docIndex: number) => {
      const set = overlapIndex.get(docIndex);
      if (!set) return [];
      return [...set].filter((k) => k !== method).map((k) => labelByKey.get(k) ?? k);
    };
  }

  const overlapCount = useMemo(() => {
    let count = 0;
    for (const set of overlapIndex.values()) if (set.size > 1) count++;
    return count;
  }, [overlapIndex]);

  // Skeleton columns shown before the first response arrives.
  const pendingColumns = [
    ...selected.map((key) => {
      const m = methods.find((x) => x.key === key);
      return { key, label: m?.label ?? key, color: methodColor(m ?? { key }) };
    }),
    ...(aspect !== "all" ? [{ key: "aspect", label: `Aspect: ${aspect}`, color: methodColor({ key: "aspect" }) }] : []),
  ];

  const hasSearched = results !== null || error !== null;
  const columnCount = results ? results.methods.length : pendingColumns.length;
  const gridClass = `grid grid-cols-1 gap-6 ${GRID_COLS[Math.min(Math.max(columnCount, 1), 5)]}`;

  return (
    <>
      <div className={hasSearched ? "mb-8" : "mb-8 mt-4 sm:mt-10"}>
        {!hasSearched && (
          <div className="mb-8 text-center">
            <h1 className="font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
              Search {health?.corpus_size ?? 727} papers, {methods.length > 0 ? `${methods.length} ways` : "many ways"}.
            </h1>
            <p className="mx-auto mt-3 max-w-xl text-[15px] text-ink-muted">
              Ask one question and compare up to four methods side by side: keyword matching, word vectors,
              transformers, and a hybrid that combines them.
            </p>
          </div>
        )}

        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          <SearchBar
            value={query}
            onChange={setQuery}
            onSubmit={() => executeSearch(query, selected, aspect)}
            loading={loading}
          />
          {methods.length > 0 && (
            <ModelPicker methods={methods} selected={selected} onChange={handleMethodsChange} disabled={loading} />
          )}
          {aspects.length > 1 && (
            <AspectSelector
              aspects={aspects}
              coverage={health?.aspect_coverage ?? {}}
              value={aspect}
              onChange={handleAspectChange}
              disabled={loading}
            />
          )}
          <ExampleQueries onPick={handleExamplePick} disabled={loading} />
        </div>
      </div>

      {error && (
        <div className="mb-8">
          <ErrorBanner message={error} />
        </div>
      )}

      {!hasSearched && !loading && <EmptyState methods={methods} corpusSize={health?.corpus_size ?? null} />}

      {loading && !results && (
        <div className={gridClass}>
          {pendingColumns.map((c) => (
            <LoadingColumn key={c.key} label={c.label} color={c.color} />
          ))}
        </div>
      )}

      {results && (
        <div className={`animate-fade-up transition-opacity ${loading ? "opacity-50" : ""}`} aria-busy={loading}>
          <div className="mb-5 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-faint">
            <p>
              Results for <span className="font-medium text-ink">&ldquo;{results.query}&rdquo;</span>
              {results.aspect !== "all" && <> · aspect <span className="font-medium text-ink">{results.aspect}</span></>}
            </p>
            <p>
              {results.took_ms} ms total
              {overlapCount > 0 && (
                <>
                  {" "}
                  · <span className="text-accent">{overlapCount}</span> paper{overlapCount === 1 ? "" : "s"} found by
                  more than one method
                </>
              )}
            </p>
          </div>
          <div className={gridClass}>
            {results.methods.map((method) => (
              <MethodColumn key={method.key} method={method} overlapLabels={overlapLabelsFor(method.key)} />
            ))}
          </div>
        </div>
      )}
    </>
  );
}
