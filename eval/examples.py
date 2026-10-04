"""Find and publish the worked examples (P3 brief, step 5).

    python eval/examples.py candidates     # list candidate queries from results/raw/
    python eval/examples.py show <qid>     # print one query with its relevant abstract and all ranks
    python eval/examples.py write          # put eval/data/examples_selected.json into summary.json

Candidates (searched on the paraphrased set first, then the exact set):
  (a) lexical methods rank the paper > 20 (or miss it) while BGE ranks it 1-3
  (b) the opposite: BM25 ranks it 1-3 while BGE ranks it > 20 (or misses it)
  (c) re-ranking changes BGE's result: hybrid fixes a BGE miss, or makes a BGE hit worse

The explanations in examples_selected.json are written by hand after reading the
query and the abstract; `write` only attaches the ranks from the raw results.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, QUERY_FILES, RESULTS_DIR, load_corpus_for, read_jsonl  # noqa: E402

from searchers.registry import METHOD_ORDER  # noqa: E402

SELECTED = DATA_DIR / "examples_selected.json"
FAR = 10**6  # stand-in for "not in the top 100"


def load_ranks() -> dict[str, dict[str, dict[str, int | None]]]:
    """{query_set: {qid: {method: rank}}}"""
    ranks: dict[str, dict[str, dict[str, int | None]]] = {}
    for path in (RESULTS_DIR / "raw").glob("*__*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for qid, q in raw["queries"].items():
            ranks.setdefault(raw["query_set"], {}).setdefault(qid, {})[raw["method"]] = q["rank"]
    return ranks


def r(value: int | None) -> int:
    return FAR if value is None else value


def fmt(value: int | None) -> str:
    return "–" if value is None else str(value)


def candidates(ranks_for_set: dict[str, dict[str, int | None]]) -> dict[str, list[str]]:
    a, b, c = [], [], []
    for qid, rk in ranks_for_set.items():
        lexical = [rk[m] for m in ("tfidf", "bm25") if m in rk]
        if not lexical or "bge" not in rk:
            continue
        if all(r(x) > 20 for x in lexical) and r(rk["bge"]) <= 3:
            a.append(qid)
        if r(rk.get("bm25")) <= 3 and r(rk["bge"]) > 20:
            b.append(qid)
        if "hybrid" in rk:
            if r(rk["bge"]) > 3 and r(rk["hybrid"]) == 1:
                c.append(qid)
            elif r(rk["bge"]) == 1 and r(rk["hybrid"]) > 5:
                c.append(qid)
    return {"a_semantic_wins": a, "b_bm25_wins": b, "c_rerank_changes_bge": c}


def query_index() -> dict[str, dict]:
    index = {}
    for name in ("exact", "paraphrased", "handwritten"):
        if QUERY_FILES[name].exists():
            for q in read_jsonl(QUERY_FILES[name]):
                index[q["qid"]] = {**q, "query_set": name}
    return index


def cmd_candidates() -> None:
    ranks = load_ranks()
    index = query_index()
    for name in ("paraphrased", "exact", "handwritten"):
        if name not in ranks:
            continue
        found = candidates(ranks[name])
        print(f"\n===== {name} =====")
        for kind, qids in found.items():
            print(f"\n{kind}: {len(qids)} candidates")
            for qid in qids[:12]:
                rk = ranks[name][qid]
                text = index[qid]["query"]
                shown = {m: fmt(rk.get(m)) for m in METHOD_ORDER if m in rk}
                print(f"  {qid} [{index[qid].get('aspect', '')}] overlap={index[qid].get('overlap')}  {text[:90]!r}\n"
                      f"      {shown}")


def cmd_show(qid: str) -> None:
    ranks, index = load_ranks(), query_index()
    q = index[qid]
    rk = ranks[q["query_set"]][qid]
    corpus = load_corpus_for(q["corpus"])
    print(f"{qid} ({q['query_set']}, {q['corpus']} corpus, overlap {q.get('overlap')})")
    print(f"query: {q['query']!r}")
    if "original" in q:
        print(f"original span: {q['original']!r}")
    print("ranks:", {m: fmt(rk.get(m)) for m in METHOD_ORDER if m in rk})
    for d in q["relevant_docs"]:
        print(f"\nrelevant doc {d} ({q['corpus']} text):\n" + textwrap.fill(corpus[d], 110))


def cmd_write() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    ranks, index = load_ranks(), query_index()
    examples = []
    for item in selected:
        q = index[item["qid"]]
        rk = ranks[q["query_set"]][item["qid"]]
        examples.append({
            "qid": item["qid"], "query": q["query"], "query_set": q["query_set"],
            "relevant_docs": q["relevant_docs"],
            "ranks": {m: rk[m] for m in METHOD_ORDER if m in rk},
            "explanation": item["explanation"],
        })
    path = RESULTS_DIR / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["examples"] = examples
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "candidates"
    if cmd == "candidates":
        cmd_candidates()
    elif cmd == "show" and len(sys.argv) > 2:
        cmd_show(sys.argv[2])
    elif cmd == "write":
        cmd_write()
    else:
        sys.exit(__doc__)
