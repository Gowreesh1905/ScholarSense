---
title: ScholarSense
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Compare 8 search methods on research abstracts
---

# ScholarSense

A comparative search engine for research-paper abstracts (Scholar Inbox dataset): type a
question and compare up to four retrieval methods side by side, from keyword matching
(TF-IDF, BM25) and word vectors (Word2Vec, GloVe) to transformers (SPECTER, BGE) and a hybrid
that fuses BM25 with BGE and re-ranks with a cross-encoder.

- **Search**: the 727 Scholar Inbox papers, or 100,727 papers (plus 100,000 arXiv
  computer-science abstracts, indexed on a cluster of three laptops with Dask).
- **Results**: MRR@10 / Recall@10 for every method on exact and back-translated queries,
  the hybrid ablation, results at 100× scale, and the laptop-cluster scaling experiment.

Runs on the free CPU tier, so the first visit after a while wakes the Space up (a few
minutes), and Hybrid + re-rank takes a few seconds per query on CPU.

Source code: https://github.com/Gowreesh1905/ScholarSense
