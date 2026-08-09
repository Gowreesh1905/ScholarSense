import type { MethodKey } from "../lib/api";
import { METHOD_STYLES } from "../lib/methodStyles";

interface ScoreBarProps {
  score: number;
  method: MethodKey;
}

export function ScoreBar({ score, method }: ScoreBarProps) {
  const displayPct = score * 100;
  const barWidth = Math.max(0, Math.min(100, displayPct));
  const style = METHOD_STYLES[method];

  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line">
        <div
          className={`h-full rounded-full ${style.bar} transition-[width] duration-500 ease-out`}
          style={{ width: `${barWidth}%` }}
        />
      </div>
      <span className={`w-12 shrink-0 text-right font-mono text-[11px] tabular-nums ${style.text}`}>
        {displayPct.toFixed(1)}%
      </span>
    </div>
  );
}
