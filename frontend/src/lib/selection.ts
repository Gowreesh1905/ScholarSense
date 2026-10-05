import type { MethodKey } from "./api";

export const MAX_METHODS = 4;
export const HYBRID_KEY = "hybrid";

/** Turn the hybrid method on/off; when 4 are already picked, it replaces the last non-hybrid pick. */
export function toggleHybrid(selected: MethodKey[], on: boolean): MethodKey[] {
  if (!on) return selected.length > 1 ? selected.filter((k) => k !== HYBRID_KEY) : selected;
  if (selected.includes(HYBRID_KEY)) return selected;
  const kept = selected.length >= MAX_METHODS ? selected.slice(0, MAX_METHODS - 1) : selected;
  return [...kept, HYBRID_KEY];
}
