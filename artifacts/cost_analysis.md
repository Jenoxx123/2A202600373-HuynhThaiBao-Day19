# Cost & Time Analysis (LAB Day 19)

## 1) Build Graph Cost
- Corpus sentences: **30**
- Raw triples: **73**
- Deduplicated triples: **73**
- Graph size: **69 nodes / 73 edges**
- Build time: **75.846 seconds**
- Build token usage: **prompt=3861, completion=3247, total=7108**
- NodeRAG status: **fallback**

## 2) Benchmark Cost (20 Questions)
- Question count: **20**
- Total eval time: **71.172 seconds**
- Flat RAG token usage: **prompt=3378, completion=387, total=3765**
- GraphRAG token usage: **prompt=5267, completion=544, total=5811**

## 3) Accuracy Comparison
- Flat RAG: **90.00%** (18/20)
- GraphRAG: **90.00%** (18/20)
- Cases Flat RAG failed but GraphRAG correct: **1**
- Cases GraphRAG failed but Flat RAG correct: **1**

## 4) Short Conclusion
- In this run, GraphRAG did not outperform Flat RAG overall (both 90%).
- GraphRAG consumed more total tokens in the benchmark run (including entity extraction step).
- GraphRAG still showed value in some multi-hop cases (at least one case where Flat RAG hallucinated but GraphRAG was correct).