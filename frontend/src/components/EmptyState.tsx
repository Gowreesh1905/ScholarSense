import type { MethodKey } from "../lib/api";
import { METHOD_STYLES } from "../lib/methodStyles";

const CARDS: { key: MethodKey; title: string; type: string; body: string }[] = [
  {
    key: "tfidf",
    title: "TF-IDF",
    type: "Lexical / keyword",
    body: "Represents text as a sparse bag-of-words weighted by rarity. Only exact word matches score — \"car\" will never match \"automobile.\"",
  },
  {
    key: "word2vec",
    title: "Word2Vec",
    type: "Static semantic",
    body: "Learns one dense vector per word from co-occurrence. Words used in similar contexts get similar vectors; documents are the average of their word vectors.",
  },
  {
    key: "bert",
    title: "BERT / SPECTER",
    type: "Contextual semantic",
    body: "A transformer that encodes full sentence context, fine-tuned specifically on scientific papers. The same word can mean different things in different sentences.",
  },
];

export function EmptyState() {
  return (
    <div className="mx-auto mt-10 max-w-4xl animate-fade-up">
      <p className="mb-5 text-center text-xs uppercase tracking-wider text-ink-faint">
        Three retrieval methods, side by side
      </p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {CARDS.map((c) => {
          const style = METHOD_STYLES[c.key];
          return (
            <div key={c.key} className={`rounded-2xl border ${style.border} ${style.bgSoft} p-5`}>
              <div className="mb-2 flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${style.dot}`} />
                <h3 className="font-display text-base font-semibold text-ink">{c.title}</h3>
              </div>
              <p className={`mb-2 text-[11px] font-medium uppercase tracking-wide ${style.text}`}>{c.type}</p>
              <p className="text-[13px] leading-relaxed text-ink-muted">{c.body}</p>
            </div>
          );
        })}
      </div>
      <p className="mt-6 text-center text-sm text-ink-faint">
        Type a query above to see how each method ranks the same 727 papers — differently.
      </p>
    </div>
  );
}
