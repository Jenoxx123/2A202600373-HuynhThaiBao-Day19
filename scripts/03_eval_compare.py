import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd

from common import (
    chat_text,
    ensure_dir,
    extract_main_entity,
    keyword_score,
    load_graph_bundle,
    load_openai_client,
    textualize_evidence,
    two_hop_evidence,
)


def read_corpus_lines(corpus_path: Path) -> List[str]:
    rows: List[str] = []
    for raw in corpus_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rows.append(line)
    return rows


def embed_texts(client, model: str, texts: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    resp = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
    faiss.normalize_L2(vectors)
    usage = {
        "prompt_tokens": int(getattr(resp.usage, "prompt_tokens", 0) or 0),
        "completion_tokens": 0,
        "total_tokens": int(getattr(resp.usage, "total_tokens", 0) or 0),
    }
    return vectors, usage


def flat_answer_faiss(
    client,
    question: str,
    model_lite: str,
    embed_model: str,
    corpus_lines: List[str],
    corpus_index: faiss.Index,
    top_k: int = 6,
) -> Tuple[str, Dict[str, int], List[str]]:
    q_vec, emb_usage = embed_texts(client, embed_model, [question])
    scores, indices = corpus_index.search(q_vec, top_k)
    retrieved = []
    for idx in indices[0]:
        if idx < 0:
            continue
        retrieved.append(corpus_lines[int(idx)])
    context = "\n".join(f"- {line}" for line in retrieved)

    system_prompt = (
        "You are a flat-RAG QA assistant. Use only the retrieved context. "
        "If missing facts, say insufficient context."
    )
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context}"
    answer, usage = chat_text(
        client=client,
        model=model_lite,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=350,
    )
    token_usage = {
        "prompt_tokens": emb_usage["prompt_tokens"] + usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": emb_usage["total_tokens"] + usage.total_tokens,
    }
    return answer, token_usage, retrieved


def graph_answer(
    client,
    question: str,
    graph,
    model_lite: str,
    model_strong: str,
) -> Tuple[str, Dict[str, int], str, List[dict]]:
    seed_entity, _, entity_usage = extract_main_entity(
        client=client,
        question=question,
        candidates=list(graph.nodes()),
        model_lite=model_lite,
        model_strong=model_strong,
    )
    evidence = two_hop_evidence(graph, seed_entity, max_edges=80) if seed_entity else []
    context = textualize_evidence(evidence)
    if not context:
        return (
            "I could not find enough graph evidence to answer.",
            {
                "prompt_tokens": entity_usage.prompt_tokens,
                "completion_tokens": entity_usage.completion_tokens,
                "total_tokens": entity_usage.total_tokens,
            },
            seed_entity,
            evidence,
        )

    system_prompt = (
        "You are a graph-RAG QA assistant. Answer only with provided evidence. "
        "If evidence is weak, clearly state uncertainty."
    )
    user_prompt = f"Question: {question}\n\n2-hop graph evidence:\n{context}"
    answer, usage = chat_text(
        client=client,
        model=model_lite,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=350,
    )
    token_usage = {
        "prompt_tokens": entity_usage.prompt_tokens + usage.prompt_tokens,
        "completion_tokens": entity_usage.completion_tokens + usage.completion_tokens,
        "total_tokens": entity_usage.total_tokens + usage.total_tokens,
    }
    return answer, token_usage, seed_entity, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flat RAG and GraphRAG on benchmarks.")
    parser.add_argument("--questions-path", default="data/benchmark_questions.csv")
    parser.add_argument("--corpus-path", default="data/corpus_tech_companies.txt")
    parser.add_argument("--graph-index-dir", default="artifacts/graph_index")
    parser.add_argument("--output-csv", default="artifacts/eval_results.csv")
    parser.add_argument("--summary-path", default="artifacts/eval_summary.json")
    parser.add_argument("--model-lite", default="gpt-4o-mini")
    parser.add_argument("--model-strong", default="gpt-4.1")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    ensure_dir(Path(args.output_csv).parent)
    client = load_openai_client()
    graph = load_graph_bundle(Path(args.graph_index_dir))
    corpus_lines = read_corpus_lines(Path(args.corpus_path))
    questions_df = pd.read_csv(args.questions_path)

    corpus_vectors, corpus_embed_usage = embed_texts(client, args.embedding_model, corpus_lines)
    dim = corpus_vectors.shape[1]
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(corpus_vectors)

    rows = []
    flat_token_total = {
        "prompt_tokens": corpus_embed_usage["prompt_tokens"],
        "completion_tokens": 0,
        "total_tokens": corpus_embed_usage["total_tokens"],
    }
    graph_token_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    t0 = time.perf_counter()
    for _, item in questions_df.iterrows():
        question = str(item["question"])
        expected = str(item["expected"])
        keywords = str(item["expected_keywords"])

        flat_start = time.perf_counter()
        flat_answer, flat_usage, flat_retrieved = flat_answer_faiss(
            client=client,
            question=question,
            model_lite=args.model_lite,
            embed_model=args.embedding_model,
            corpus_lines=corpus_lines,
            corpus_index=flat_index,
            top_k=args.top_k,
        )
        flat_seconds = round(time.perf_counter() - flat_start, 3)

        graph_start = time.perf_counter()
        graph_resp, graph_usage, seed_entity, graph_evidence = graph_answer(
            client=client,
            question=question,
            graph=graph,
            model_lite=args.model_lite,
            model_strong=args.model_strong,
        )
        graph_seconds = round(time.perf_counter() - graph_start, 3)

        flat_hit, flat_total, flat_correct = keyword_score(flat_answer, keywords)
        graph_hit, graph_total, graph_correct = keyword_score(graph_resp, keywords)

        if (not flat_correct) and graph_correct:
            note = "Flat RAG miss/hallucination; GraphRAG correct."
        elif flat_correct and (not graph_correct):
            note = "GraphRAG underperformed on this question."
        elif flat_correct and graph_correct:
            note = "Both correct."
        else:
            note = "Both incorrect or incomplete."

        flat_token_total["prompt_tokens"] += flat_usage["prompt_tokens"]
        flat_token_total["completion_tokens"] += flat_usage["completion_tokens"]
        flat_token_total["total_tokens"] += flat_usage["total_tokens"]

        graph_token_total["prompt_tokens"] += graph_usage["prompt_tokens"]
        graph_token_total["completion_tokens"] += graph_usage["completion_tokens"]
        graph_token_total["total_tokens"] += graph_usage["total_tokens"]

        rows.append(
            {
                "question": question,
                "expected": expected,
                "expected_keywords": keywords,
                "flat_answer": flat_answer,
                "graph_answer": graph_resp,
                "flat_hit_keywords": f"{flat_hit}/{flat_total}",
                "graph_hit_keywords": f"{graph_hit}/{graph_total}",
                "flat_correct": int(flat_correct),
                "graph_correct": int(graph_correct),
                "hallucination_note": note,
                "flat_seconds": flat_seconds,
                "graph_seconds": graph_seconds,
                "graph_seed_entity": seed_entity,
                "flat_retrieval_preview": " | ".join(flat_retrieved[:3]),
                "graph_evidence_count": len(graph_evidence),
            }
        )

    elapsed = round(time.perf_counter() - t0, 3)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    summary = {
        "question_count": int(len(result_df)),
        "flat_correct_count": int(result_df["flat_correct"].sum()),
        "graph_correct_count": int(result_df["graph_correct"].sum()),
        "flat_accuracy": round(float(result_df["flat_correct"].mean()), 4),
        "graph_accuracy": round(float(result_df["graph_correct"].mean()), 4),
        "flat_token_usage": flat_token_total,
        "graph_token_usage": graph_token_total,
        "total_eval_seconds": elapsed,
    }
    Path(args.summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
