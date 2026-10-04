import type { MethodInfo, MethodKey } from "../lib/api";
import { FAMILY_LABELS, FAMILY_ORDER, methodColor, tint } from "../lib/methodStyles";
import { HYBRID_KEY, MAX_METHODS, toggleHybrid } from "../lib/selection";

interface ModelPickerProps {
  methods: MethodInfo[];
  selected: MethodKey[];
  onChange: (next: MethodKey[]) => void;
  disabled?: boolean;
}

export function ModelPicker({ methods, selected, onChange, disabled }: ModelPickerProps) {
  const atMax = selected.length >= MAX_METHODS;
  const hasHybrid = methods.some((m) => m.key === HYBRID_KEY);
  const hybridOn = selected.includes(HYBRID_KEY);

  function toggle(key: MethodKey) {
    if (selected.includes(key)) {
      if (selected.length > 1) onChange(selected.filter((k) => k !== key));
    } else if (!atMax) {
      onChange([...selected, key]);
    }
  }

  const groups = FAMILY_ORDER.map((family) => ({
    family,
    methods: methods.filter((m) => m.family === family),
  })).filter((g) => g.methods.length > 0);

  return (
    <div className="rounded-2xl border border-line bg-surface p-3.5 shadow-soft sm:p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <p className="text-xs text-ink-muted">
          <span className="font-medium text-ink">Compare</span> {selected.length} of {MAX_METHODS} methods
          {atMax && <span className="text-ink-faint"> · deselect one to pick another</span>}
        </p>
        {hasHybrid && (
          <button
            type="button"
            role="switch"
            aria-checked={hybridOn}
            disabled={disabled}
            onClick={() => onChange(toggleHybrid(selected, !hybridOn))}
            className="flex items-center gap-2 text-xs font-medium text-ink-muted transition hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
          >
            <span
              className="relative h-5 w-9 rounded-full transition-colors"
              style={{ backgroundColor: hybridOn ? methodColor({ key: HYBRID_KEY }) : "var(--line-strong)" }}
            >
              <span
                className="absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-[left]"
                style={{ left: hybridOn ? "18px" : "2px" }}
              />
            </span>
            Hybrid + re-rank
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-x-4 gap-y-2.5 sm:grid-cols-2">
        {groups.map(({ family, methods: group }) => (
          <div key={family} className="min-w-0">
            <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">{FAMILY_LABELS[family]}</p>
            <div className="flex flex-wrap gap-1.5">
              {group.map((m) => {
                const color = methodColor(m);
                const isOn = selected.includes(m.key);
                const blocked = !isOn && atMax;
                return (
                  <button
                    key={m.key}
                    type="button"
                    aria-pressed={isOn}
                    disabled={disabled || blocked}
                    onClick={() => toggle(m.key)}
                    title={blocked ? `Up to ${MAX_METHODS} methods at once` : m.description}
                    className="flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition disabled:cursor-not-allowed disabled:opacity-45"
                    style={
                      isOn
                        ? { borderColor: tint(color, 0x80), backgroundColor: tint(color, 0x1a), color: "var(--ink)" }
                        : { borderColor: "var(--line)", color: "var(--ink-muted)" }
                    }
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full border"
                      style={{ borderColor: color, backgroundColor: isOn ? color : "transparent" }}
                    />
                    {m.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
