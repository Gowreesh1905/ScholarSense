interface ErrorBannerProps {
  message: string;
}

export function ErrorBanner({ message }: ErrorBannerProps) {
  return (
    <div className="mx-auto flex max-w-2xl animate-fade-up items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/8 px-4 py-3.5 text-left">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 h-4 w-4 shrink-0 text-red-500">
        <circle cx="12" cy="12" r="10" />
        <path d="M12 8v4M12 16h.01" />
      </svg>
      <div>
        <p className="text-sm font-medium text-red-500">Search failed</p>
        <p className="mt-0.5 text-[13px] text-ink-muted">{message}</p>
        <p className="mt-1 text-[12px] text-ink-faint">
          Is the API running? Try <code className="rounded bg-surface-hover px-1 py-0.5 font-mono">python backend/app.py</code> in another terminal.
        </p>
      </div>
    </div>
  );
}
