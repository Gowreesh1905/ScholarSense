import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Header } from "./components/Header";
import { SearchBar } from "./components/SearchBar";
import { ExampleQueries } from "./components/ExampleQueries";
import { MethodColumn } from "./components/MethodColumn";
import { LoadingColumn } from "./components/LoadingColumn";
import { EmptyState } from "./components/EmptyState";
import { ErrorBanner } from "./components/ErrorBanner";
import { Footer } from "./components/Footer";
import { fetchHealth, runSearch } from "./lib/api";
import type { HealthResponse, MethodKey, SearchResponse } from "./lib/api";

const METHOD_ORDER: MethodKey[] = ["tfidf", "word2vec", "bert"];

function buildOverlapIndex(response: SearchResponse) {
  const index = new Map<string, Set<MethodKey>>();
  for (const method of response.methods) {
    for (const hit of method.results) {
      if (!index.has(hit.arxiv_id)) index.set(hit.arxiv_id, new Set());
      index.get(hit.arxiv_id)!.add(method.key);
    }
  }
  return index;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const requestId = useRef(0);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealthError(true));
  }, []);

  const executeSearch = useCallback(async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;

    const thisRequest = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const response = await runSearch(trimmed, 5);
      if (thisRequest === requestId.current) {
        setResults(response);
      }
    } catch (err) {
      if (thisRequest === requestId.current) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
        setResults(null);
      }
    } finally {
      if (thisRequest === requestId.current) {
        setLoading(false);
      }
    }
  }, []);

  function handleExamplePick(q: string) {
    setQuery(q);
    executeSearch(q);
  }

  const overlapIndex = useMemo(() => (results ? buildOverlapIndex(results) : new Map()), [results]);
  const labelByKey = useMemo(
    () => new Map(results?.methods.map((m) => [m.key, m.label]) ?? []),
    [results],
  );

  function overlapLabelsFor(method: MethodKey) {
    return (arxivId: string) => {
      const set = overlapIndex.get(arxivId);
      if (!set) return [];
      return [...set].filter((k) => k !== method).map((k) => labelByKey.get(k) ?? k);
    };
  }

  const overlapCount = useMemo(() => {
    let count = 0;
    for (const set of overlapIndex.values()) if (set.size > 1) count++;
    return count;
  }, [overlapIndex]);

  const methodsByKey = useMemo(
    () => new Map(results?.methods.map((m) => [m.key, m]) ?? []),
    [results],
  );

  const hasSearched = results !== null || error !== null;

  return (
    <div className="flex min-h-screen flex-col bg-canvas">
      <Header health={health} healthError={healthError} />

      <main className="mx-auto w-full max-w-6xl flex-1 px-5 py-10 sm:px-8 sm:py-14">
        <div className={hasSearched ? "mb-10" : "mb-8 mt-6 sm:mt-12"}>
          {!hasSearched && (
            <div className="mb-8 text-center">
              <h1 className="font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
                Search 727 papers, three ways.
              </h1>
              <p className="mx-auto mt-3 max-w-xl text-[15px] text-ink-muted">
                Ask one question and watch a lexical, a static-semantic, and a contextual model each rank the corpus
                on their own terms.
              </p>
            </div>
          )}

          <div className="mx-auto max-w-2xl">
            <SearchBar
              value={query}
              onChange={setQuery}
              onSubmit={() => executeSearch(query)}
              loading={loading}
            />
            <div className="mt-4">
              <ExampleQueries onPick={handleExamplePick} disabled={loading} />
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-8">
            <ErrorBanner message={error} />
          </div>
        )}

        {!hasSearched && !loading && <EmptyState />}

        {loading && !results && (
          <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
            {METHOD_ORDER.map((key) => (
              <LoadingColumn key={key} method={key} />
            ))}
          </div>
        )}

        {results && (
          <div className="animate-fade-up">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-faint">
              <p>
                Results for <span className="font-medium text-ink">&ldquo;{results.query}&rdquo;</span>
              </p>
              <p>
                {results.took_ms} ms
                {overlapCount > 0 && (
                  <>
                    {" "}
                    · <span className="text-accent">{overlapCount}</span> paper{overlapCount === 1 ? "" : "s"} found
                    by multiple methods
                  </>
                )}
              </p>
            </div>
            <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
              {METHOD_ORDER.map((key) => {
                const method = methodsByKey.get(key);
                if (!method) return null;
                return <MethodColumn key={key} method={method} overlapLabels={overlapLabelsFor(key)} />;
              })}
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
