import { useEffect, useState } from "react";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { useHashRoute } from "./hooks/useHashRoute";
import { fetchHealth } from "./lib/api";
import type { HealthResponse } from "./lib/api";
import { ResultsPage } from "./pages/ResultsPage";
import { SearchPage } from "./pages/SearchPage";

export default function App() {
  const { route } = useHashRoute();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);

  // The API loads every model before it answers, so keep retrying while it starts up.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const attempt = () => {
      fetchHealth()
        .then((h) => {
          if (cancelled) return;
          setHealth(h);
          setHealthError(false);
        })
        .catch(() => {
          if (cancelled) return;
          setHealthError(true);
          timer = setTimeout(attempt, 3000);
        });
    };
    attempt();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return (
    <div className="flex min-h-screen flex-col bg-canvas">
      <Header health={health} healthError={healthError} route={route} />

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-10 sm:px-8 sm:py-14">
        {/* Search stays mounted so its query and results survive a visit to the Results tab. */}
        <div hidden={route !== "search"}>
          <SearchPage health={health} />
        </div>
        {route === "results" && <ResultsPage />}
      </main>

      <Footer />
    </div>
  );
}
