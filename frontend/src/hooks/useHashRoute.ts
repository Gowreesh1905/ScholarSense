import { useCallback, useEffect, useState } from "react";

export type Route = "search" | "results";

function readRoute(): Route {
  return window.location.hash === "#results" ? "results" : "search";
}

/** The current page, kept in the URL hash (#search / #results) so it is linkable and survives reload. */
export function useHashRoute() {
  const [route, setRoute] = useState<Route>(readRoute);

  useEffect(() => {
    const onChange = () => setRoute(readRoute());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((next: Route) => {
    window.location.hash = next;
  }, []);

  return { route, navigate };
}
