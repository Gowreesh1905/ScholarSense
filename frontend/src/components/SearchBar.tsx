import { useRef, type FormEvent } from "react";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
}

export function SearchBar({ value, onChange, onSubmit, loading }: SearchBarProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!value.trim() || loading) return;
    onSubmit();
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="group relative flex items-center rounded-2xl border border-line bg-surface shadow-soft transition focus-within:border-accent/50 focus-within:ring-4 focus-within:ring-accent/10">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          className="ml-4 h-5 w-5 shrink-0 text-ink-faint transition group-focus-within:text-accent sm:ml-5"
        >
          <circle cx="11" cy="11" r="7" />
          <path d="M21 21l-4.35-4.35" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Ask in plain English — e.g. automatic translation of spoken language"
          className="min-w-0 flex-1 bg-transparent px-3 py-4 text-[15px] text-ink placeholder:text-ink-faint focus:outline-none sm:text-base"
          spellCheck={false}
          autoFocus
        />
        <button
          type="submit"
          disabled={!value.trim() || loading}
          className="mr-1.5 flex h-10 shrink-0 items-center gap-2 rounded-xl bg-accent px-4 text-sm font-medium text-white transition enabled:hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 sm:mr-2 sm:h-11 sm:px-5"
        >
          {loading ? (
            <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4 animate-spin">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
              <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
          ) : (
            <span>Search</span>
          )}
        </button>
      </div>
    </form>
  );
}
