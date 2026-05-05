import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from common import (
    UsageStats,
    build_graph_from_triples,
    chat_json,
    deduplicate_triples,
    ensure_dir,
    load_openai_client,
    save_graph_bundle,
    save_jsonl,
    validate_and_normalize_triples,
)


def read_corpus_lines(corpus_path: Path) -> List[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    rows: List[str] = []
    for raw in corpus_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def extract_triples_two_stage(
    sentence: str,
    client,
    model_lite: str,
    model_strong: str,
) -> Tuple[List[dict], str, bool, UsageStats]:
    usage = UsageStats()

    system_prompt = (
        "You are an information extraction engine.\n"
        "Extract fact triples from one sentence.\n"
        "Return valid JSON only with this schema:\n"
        "{\n"
        '  "triples": [\n'
        "    {\n"
        '      "subject": "entity",\n'
        '      "relation": "UPPER_SNAKE_CASE_RELATION",\n'
        '      "object": "entity_or_value",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Keep relation names concise.\n"
        "- If no reliable fact, return an empty triples array.\n"
    )
    user_prompt = f"Sentence: {sentence}"

    data_lite, usage_lite, _ = chat_json(client, model_lite, system_prompt, user_prompt)
    usage.add(usage_lite.prompt_tokens, usage_lite.completion_tokens, usage_lite.total_tokens)
    triples_lite = validate_and_normalize_triples((data_lite or {}).get("triples", []))

    low_conf = len(triples_lite) == 0 or max(t["confidence"] for t in triples_lite) < 0.55
    if not low_conf:
        return triples_lite, model_lite, False, usage

    data_strong, usage_strong, _ = chat_json(client, model_strong, system_prompt, user_prompt)
    usage.add(
        usage_strong.prompt_tokens, usage_strong.completion_tokens, usage_strong.total_tokens
    )
    triples_strong = validate_and_normalize_triples((data_strong or {}).get("triples", []))
    if triples_strong:
        return triples_strong, model_strong, True, usage
    return triples_lite, model_lite, True, usage


def categorize_nodes(graph: nx.MultiDiGraph) -> Dict[str, str]:
    categories: Dict[str, str] = {}
    founded_by_targets = set()
    headquartered_targets = set()
    product_targets = set()
    company_like_subjects = set()

    for u, v, data in graph.edges(data=True):
        rel = str(data.get("relation", "")).upper()
        if rel == "FOUNDED_BY":
            founded_by_targets.add(v)
            company_like_subjects.add(u)
        elif rel in {"HEADQUARTERED_IN", "HQ_IN"}:
            headquartered_targets.add(v)
            company_like_subjects.add(u)
        elif rel in {"RELEASED", "INTRODUCED", "DEVELOPED_BY", "DEVELOPS"}:
            product_targets.add(v)
            company_like_subjects.add(u)
        elif rel in {"ACQUIRED", "ACQUIRED_BY", "INVESTED_IN", "MANUFACTURES_CHIPS_FOR"}:
            company_like_subjects.add(u)

    for node in graph.nodes():
        text = str(node).strip()
        if text.isdigit() and len(text) == 4:
            categories[node] = "year"
        elif node in founded_by_targets:
            categories[node] = "person"
        elif node in headquartered_targets:
            categories[node] = "location"
        elif node in product_targets:
            categories[node] = "product"
        elif node in company_like_subjects:
            categories[node] = "company"
        else:
            categories[node] = "other"
    return categories


def improved_layout(graph: nx.MultiDiGraph, categories: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    base = graph.to_undirected()
    shells = []
    shell_center = [n for n, c in categories.items() if c == "company"]
    shell_mid = [n for n, c in categories.items() if c in {"product", "location", "other"}]
    shell_outer = [n for n, c in categories.items() if c in {"person", "year"}]

    for shell in [shell_center, shell_mid, shell_outer]:
        if shell:
            shells.append(shell)
    if not shells:
        shells = [list(graph.nodes())]

    pos0 = nx.shell_layout(base, nlist=shells, scale=7.5, rotate=0)
    pos = nx.spring_layout(
        base,
        pos=pos0,
        seed=42,
        k=3.0 / max(1.0, (base.number_of_nodes() ** 0.5) / 4.0),
        iterations=520,
        scale=13.0,
    )
    pos = resolve_node_collisions(pos, min_dist=1.15, iterations=260)
    return pos


def resolve_node_collisions(
    pos: Dict[str, Tuple[float, float]],
    min_dist: float = 0.75,
    iterations: int = 160,
) -> Dict[str, Tuple[float, float]]:
    nodes = list(pos.keys())
    coord = {n: [float(pos[n][0]), float(pos[n][1])] for n in nodes}

    for _ in range(iterations):
        moved = False
        for i in range(len(nodes)):
            n1 = nodes[i]
            x1, y1 = coord[n1]
            for j in range(i + 1, len(nodes)):
                n2 = nodes[j]
                x2, y2 = coord[n2]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                if dist_sq == 0.0:
                    # Deterministic tiny nudge for exact overlap.
                    dx = ((hash(n1) % 31) - 15) * 1e-3
                    dy = ((hash(n2) % 29) - 14) * 1e-3
                    dist_sq = dx * dx + dy * dy
                dist = dist_sq ** 0.5
                if dist < min_dist:
                    moved = True
                    push = (min_dist - dist) * 0.5
                    ux = dx / dist
                    uy = dy / dist
                    coord[n1][0] += ux * push
                    coord[n1][1] += uy * push
                    coord[n2][0] -= ux * push
                    coord[n2][1] -= uy * push
        if not moved:
            break

    return {n: (coord[n][0], coord[n][1]) for n in nodes}


def unique_edge_labels(graph: nx.MultiDiGraph) -> Dict[Tuple[str, str], str]:
    labels: Dict[Tuple[str, str], List[str]] = {}
    for u, v, data in graph.edges(data=True):
        key = tuple(sorted((u, v)))
        labels.setdefault(key, [])
        rel = str(data.get("relation", "RELATED_TO"))
        if rel not in labels[key]:
            labels[key].append(rel)
    return {k: " | ".join(v) for k, v in labels.items()}


def build_priority_edge_labels(
    graph: nx.MultiDiGraph,
    categories: Dict[str, str],
    max_labels: int = 24,
) -> Dict[Tuple[str, str], str]:
    labels = unique_edge_labels(graph)
    degree = dict(graph.degree())

    def score(item: Tuple[Tuple[str, str], str]) -> float:
        (u, v), txt = item
        cat_u = categories.get(u, "other")
        cat_v = categories.get(v, "other")
        cat_score = 0.0
        if "company" in (cat_u, cat_v):
            cat_score += 2.5
        if cat_u == "company" and cat_v == "company":
            cat_score += 1.5
        if "person" in (cat_u, cat_v):
            cat_score += 0.5
        rel_score = 0.25 * len(txt.split("|"))
        deg_score = 0.04 * (degree.get(u, 0) + degree.get(v, 0))
        return cat_score + rel_score + deg_score

    ranked = sorted(labels.items(), key=score, reverse=True)
    return dict(ranked[:max_labels])


def format_node_label(node: str, max_len: int = 14) -> str:
    text = str(node).strip()
    if len(text) <= max_len:
        return text
    parts = text.split()
    if len(parts) >= 2:
        line1 = []
        line2 = []
        total = 0
        for p in parts:
            if total + len(p) + (1 if line1 else 0) <= max_len:
                line1.append(p)
                total += len(p) + (1 if line1 else 0)
            else:
                line2.append(p)
        if line2:
            second = " ".join(line2)
            if len(second) > max_len:
                second = second[: max_len - 1] + "."
            return f"{' '.join(line1)}\n{second}"
    return text[: max_len - 1] + "."


def draw_graph(graph: nx.MultiDiGraph, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    categories = categorize_nodes(graph)
    pos = improved_layout(graph, categories)

    plt.figure(figsize=(22, 14))
    color_map = {
        "company": "#4ea8de",
        "person": "#90be6d",
        "year": "#adb5bd",
        "location": "#f8961e",
        "product": "#f9844a",
        "other": "#a2d2ff",
    }
    size_map = {
        "company": 860,
        "person": 470,
        "year": 330,
        "location": 430,
        "product": 430,
        "other": 390,
    }

    for category in ["company", "person", "year", "location", "product", "other"]:
        nodes = [n for n, c in categories.items() if c == category]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_size=size_map[category],
            node_color=color_map[category],
            alpha=0.92,
            edgecolors="#184e77",
            linewidths=0.45,
        )

    label_dict = {node: format_node_label(node, max_len=14) for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=label_dict,
        font_size=6.1,
        font_weight="normal",
        font_color="#0f172a",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        width=0.95,
        edge_color="#5f0f40",
        alpha=0.42,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=9,
        connectionstyle="arc3,rad=0.08",
    )
    edge_labels = build_priority_edge_labels(graph, categories, max_labels=24)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=4.7,
        rotate=False,
        font_color="#202020",
        label_pos=0.45,
        bbox={"boxstyle": "round,pad=0.12", "fc": (1, 1, 1, 0.70), "ec": (0, 0, 0, 0.06)},
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=260)
    plt.close()


def build_noderag_workspace(
    corpus_lines: List[str],
    graph_index_dir: Path,
    model_for_noderag: str,
    graph_node_count: int,
    graph_edge_count: int,
    deduped_triples_count: int,
) -> Dict[str, str]:
    status: Dict[str, str] = {"status": "skipped", "message": "not requested"}
    workspace = graph_index_dir / "noderag_workspace"
    input_dir = workspace / "input"
    ensure_dir(input_dir)
    (input_dir / "corpus.txt").write_text("\n".join(corpus_lines), encoding="utf-8")

    config_payload = {
        "config": {
            "main_folder": str(workspace.resolve()),
            "language": "English",
            "chunk_size": 800,
            "docu_type": "mixed",
            "dim": 1536,
        },
        "model_config": {
            "service_provider": "openai",
            "model_name": model_for_noderag,
            "temperature": 0.0,
            "max_tokens": 1000,
        },
        "embedding_config": {
            "service_provider": "openai_embedding",
            "embedding_model_name": "text-embedding-3-small",
        },
    }
    (workspace / "Node_config.generated.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        from NodeRAG import NodeConfig, NodeRag

        # Avoid Windows cp1252 crashes when NodeRAG prints Unicode symbols.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

        node_cfg = NodeConfig(config_payload)
        pipeline = NodeRag(node_cfg, web_ui=True)
        pipeline.run()
        status = {"status": "ok", "message": "NodeRAG pipeline completed"}
    except Exception as exc:  # noqa: BLE001
        fallback_payload = {
            "reason": str(exc),
            "fallback_engine": "custom_networkx_graph",
            "graph_node_count": graph_node_count,
            "graph_edge_count": graph_edge_count,
            "deduplicated_triples": deduped_triples_count,
            "note": (
                "NodeRAG internal pipeline failed, but GraphRAG artifacts are available from "
                "scripts/01_build_graph.py output (triples.jsonl + graph.pkl + graph.png)."
            ),
        }
        (workspace / "noderag_fallback.json").write_text(
            json.dumps(fallback_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        status = {
            "status": "fallback",
            "message": f"NodeRAG failed and fallback metadata was created: {exc}",
        }

    (workspace / "noderag_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GraphRAG index from tech corpus.")
    parser.add_argument("--corpus-path", default="data/corpus_tech_companies.txt")
    parser.add_argument("--triples-path", default="artifacts/triples.jsonl")
    parser.add_argument("--graph-index-dir", default="artifacts/graph_index")
    parser.add_argument("--graph-image-path", default="artifacts/graph.png")
    parser.add_argument("--stats-path", default="artifacts/build_stats.json")
    parser.add_argument("--model-lite", default="gpt-4o-mini")
    parser.add_argument("--model-strong", default="gpt-4.1")
    parser.add_argument("--run-noderag", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    client = load_openai_client()

    corpus_path = Path(args.corpus_path)
    triples_path = Path(args.triples_path)
    graph_index_dir = Path(args.graph_index_dir)
    graph_image_path = Path(args.graph_image_path)
    stats_path = Path(args.stats_path)

    ensure_dir(graph_index_dir)
    ensure_dir(triples_path.parent)

    corpus_lines = read_corpus_lines(corpus_path)
    all_triples: List[dict] = []
    usage_total = UsageStats()
    escalated_count = 0

    for i, sentence in enumerate(corpus_lines, start=1):
        triples, model_used, escalated, usage = extract_triples_two_stage(
            sentence=sentence,
            client=client,
            model_lite=args.model_lite,
            model_strong=args.model_strong,
        )
        usage_total.add(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        if escalated:
            escalated_count += 1

        for triple in triples:
            triple["source_sentence_id"] = i
            triple["source_text"] = sentence
            triple["model_used"] = model_used
            all_triples.append(triple)

    deduped = deduplicate_triples(all_triples)
    save_jsonl(triples_path, deduped)

    graph = build_graph_from_triples(deduped)
    save_graph_bundle(graph, graph_index_dir)
    draw_graph(graph, graph_image_path)

    noderag_status = {"status": "skipped", "message": "use --run-noderag to execute"}
    if args.run_noderag:
        noderag_status = build_noderag_workspace(
            corpus_lines=corpus_lines,
            graph_index_dir=graph_index_dir,
            model_for_noderag=args.model_lite,
            graph_node_count=graph.number_of_nodes(),
            graph_edge_count=graph.number_of_edges(),
            deduped_triples_count=len(deduped),
        )

    elapsed = round(time.perf_counter() - start, 3)
    stats = {
        "corpus_sentences": len(corpus_lines),
        "raw_triples": len(all_triples),
        "deduplicated_triples": len(deduped),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "escalated_items": escalated_count,
        "token_usage": {
            "prompt_tokens": usage_total.prompt_tokens,
            "completion_tokens": usage_total.completion_tokens,
            "total_tokens": usage_total.total_tokens,
        },
        "build_seconds": elapsed,
        "noderag_status": noderag_status,
    }
    ensure_dir(stats_path.parent)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
