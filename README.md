# LAB Day 19 - GraphRAG vs Flat RAG

Project này triển khai pipeline GraphRAG cho domain công ty công nghệ, gồm:
- Build triples + graph
- Query theo 2-hop graph retrieval
- Benchmark 20 câu hỏi giữa Flat RAG và GraphRAG

## 1) Yêu cầu

- Python 3.10+ (khuyến nghị 3.11)
- Windows PowerShell (hoặc terminal tương đương)
- OpenAI API key

## 2) Cài đặt

### Bước 1: Tạo và kích hoạt `venv` (nếu chưa có)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Bước 2: Cài thư viện

```powershell
pip install -r requirements.txt
```

### Bước 3: Tạo file `.env`

Tạo file `.env` ở root project:

```env
OPENAI_API_KEY=sk-...
```

## 3) Cấu trúc chính

- `scripts/01_build_graph.py`: trích xuất triples + build graph + xuất ảnh graph
- `scripts/02_query_graphrag.py`: hỏi đáp GraphRAG theo 2-hop evidence
- `scripts/03_eval_compare.py`: benchmark 20 câu Flat RAG vs GraphRAG
- `data/corpus_tech_companies.txt`: corpus mẫu
- `data/benchmark_questions.csv`: bộ 20 câu benchmark

## 4) Cách chạy

### 4.1 Build graph

```powershell
.\venv\Scripts\python.exe scripts/01_build_graph.py --run-noderag
```

Output chính:
- `artifacts/triples.jsonl`
- `artifacts/graph.png`
- `artifacts/graph_index/graph.pkl`
- `artifacts/build_stats.json`

### 4.2 Query GraphRAG

```powershell
.\venv\Scripts\python.exe scripts/02_query_graphrag.py --question "Which company acquired YouTube and who founded that company?"
```

Script sẽ in JSON gồm:
- `answer`
- `used_entities`
- `2hop_evidence`
- `token_usage`

### 4.3 Chạy benchmark 20 câu

```powershell
.\venv\Scripts\python.exe scripts/03_eval_compare.py
```

Output:
- `artifacts/eval_results.csv`
- `artifacts/eval_summary.json`

## 5) File nộp bài (Deliverables)

Bạn đã có đủ bộ nộp nếu các file sau tồn tại:
- Mã nguồn: `scripts/*.py`
- Ảnh graph: `artifacts/graph.png`
- Bảng benchmark 20 câu: `artifacts/eval_results.csv`
- Phân tích chi phí/time: `artifacts/cost_analysis.md`

## 6) Lỗi thường gặp

### `openai.APIConnectionError`
- Kiểm tra mạng/proxy/firewall.
- Chạy lại lệnh benchmark/build khi mạng ổn định.

### NodeRAG fail trong quá trình `--run-noderag`
- Script hiện có fallback và vẫn sinh đầy đủ artifacts graph custom.
- Kiểm tra trạng thái tại:
  - `artifacts/build_stats.json` (`noderag_status`)
  - `artifacts/graph_index/noderag_workspace/noderag_fallback.json`

## 7) Lệnh nhanh (copy chạy)

```powershell
.\venv\Scripts\python.exe scripts/01_build_graph.py --run-noderag
.\venv\Scripts\python.exe scripts/02_query_graphrag.py --question "Who founded OpenAI?"
.\venv\Scripts\python.exe scripts/03_eval_compare.py
```

