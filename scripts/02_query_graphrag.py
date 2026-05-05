import argparse
import json
from pathlib import Path

from common import (
    chat_text,
    extract_main_entity,
    load_graph_bundle,
    load_openai_client,
    textualize_evidence,
    two_hop_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the GraphRAG index with 2-hop retrieval.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--graph-index-dir", default="artifacts/graph_index")
    parser.add_argument("--model-lite", default="gpt-4o-mini")
    parser.add_argument("--model-strong", default="gpt-4.1")
    parser.add_argument("--max-evidence-edges", type=int, default=80)
    args = parser.parse_args()

    client = load_openai_client()
    graph = load_graph_bundle(Path(args.graph_index_dir))
    candidates = list(graph.nodes())

    seed_entity, entity_model, entity_usage = extract_main_entity(
        client=client,
        question=args.question,
        candidates=candidates,
        model_lite=args.model_lite,
        model_strong=args.model_strong,
    )

    evidence = two_hop_evidence(graph, seed_entity, max_edges=args.max_evidence_edges) if seed_entity else []
    context = textualize_evidence(evidence)

    if not context:
        answer_text = "I could not find relevant evidence in the graph for this question."
        llm_usage = {
            "prompt_tokens": entity_usage.prompt_tokens,
            "completion_tokens": entity_usage.completion_tokens,
            "total_tokens": entity_usage.total_tokens,
        }
        answer_model = "none"
    else:
        system_prompt = (
            "You are a QA assistant that must answer only from provided graph evidence. "
            "If evidence is insufficient, say so clearly."
        )
        user_prompt = (
            f"Question: {args.question}\n\n"
            f"Graph evidence (2-hop):\n{context}\n\n"
            "Write a concise answer in 2-4 sentences."
        )
        answer_text, usage = chat_text(
            client=client,
            model=args.model_lite,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=350,
        )
        llm_usage = {
            "prompt_tokens": entity_usage.prompt_tokens + usage.prompt_tokens,
            "completion_tokens": entity_usage.completion_tokens + usage.completion_tokens,
            "total_tokens": entity_usage.total_tokens + usage.total_tokens,
        }
        answer_model = args.model_lite

    payload = {
        "question": args.question,
        "answer": answer_text,
        "used_entities": [seed_entity] if seed_entity else [],
        "entity_extraction_model": entity_model,
        "answer_model": answer_model,
        "2hop_evidence": evidence,
        "token_usage": llm_usage,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
