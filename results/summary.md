| Method | exact MRR@10 | exact R@10 | paraphrased MRR@10 | paraphrased R@10 | handwritten MRR@10 | handwritten R@10 | Latency mean (ms) | Latency p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.519 | 0.710 | 0.438 | 0.647 | 0.400 | 0.453 | 2.1 | 3.3 |
| BM25 | 0.523 | 0.704 | 0.439 | 0.624 | 0.407 | 0.503 | 2.5 | 5.8 |
| Word2Vec (corpus-trained) | 0.272 | 0.433 | 0.215 | 0.362 | 0.408 | 0.324 | 0.9 | 1.4 |
| GloVe 300d (pretrained) | 0.237 | 0.409 | 0.208 | 0.363 | 0.379 | 0.358 | **0.6** | **0.8** |
| SPECTER | 0.233 | 0.395 | 0.197 | 0.346 | 0.387 | 0.418 | 9.4 | 11.0 |
| BGE-base | 0.498 | 0.711 | 0.462 | 0.666 | **0.900** | **0.868** | 10.6 | 18.2 |
| BM25 + BGE (RRF) | **0.560** | 0.757 | **0.507** | 0.708 | 0.738 | 0.719 | 17.2 | 34.7 |
| Hybrid + re-rank | 0.548 | **0.766** | 0.506 | **0.717** | 0.818 | 0.750 | 150.5 | 200.7 |
| BM25 + query expansion | 0.411 | 0.702 | 0.348 | 0.612 | 0.498 | 0.525 | – | – |

Corpus: 727 abstracts. Device for latency: cuda (NVIDIA GeForce RTX 4050 Laptop GPU).

Query sets: **exact** n=1030 (masked corpus); **paraphrased** n=1030 (masked corpus); **handwritten** n=20 (full corpus).

Bold = best value in the column (highest accuracy, lowest latency). Latency: single-query `search()` on the full corpus, mean and p95 over 200 queries after 5 warm-up queries.
