import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.total_tokens += int(total or 0)


def load_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env before running.")
    return OpenAI(api_key=api_key)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def strip_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def safe_json_loads(text: str) -> Optional[dict]:
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> Tuple[Optional[dict], UsageStats, str]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = UsageStats()
    usage.add(
        getattr(resp.usage, "prompt_tokens", 0),
        getattr(resp.usage, "completion_tokens", 0),
        getattr(resp.usage, "total_tokens", 0),
    )
    content = (resp.choices[0].message.content or "").strip()
    return safe_json_loads(content), usage, content


def chat_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> Tuple[str, UsageStats]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = UsageStats()
    usage.add(
        getattr(resp.usage, "prompt_tokens", 0),
        getattr(resp.usage, "completion_tokens", 0),
        getattr(resp.usage, "total_tokens", 0),
    )
    return (resp.choices[0].message.content or "").strip(), usage


def normalize_relation(value: str) -> str:
    rel = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper())
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel or "RELATED_TO"


def normalize_entity(value: str) -> str:
    entity = re.sub(r"\s+", " ", value.strip())
    return entity


def validate_and_normalize_triples(raw: Sequence[dict]) -> List[dict]:
    triples: List[dict] = []
    for item in raw:
        subject = normalize_entity(str(item.get("subject", "")))
        relation = normalize_relation(str(item.get("relation", "")))
        obj = normalize_entity(str(item.get("object", "")))
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if not subject or not obj:
            continue
        triples.append(
            {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "confidence": round(confidence, 4),
            }
        )
    return triples


def deduplicate_triples(rows: Sequence[dict]) -> List[dict]:
    seen = set()
    output: List[dict] = []
    for row in rows:
        key = (row["subject"].lower(), row["relation"].upper(), row["object"].lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def save_jsonl(path: Path, rows: Sequence[dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_graph_from_triples(triples: Sequence[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for idx, row in enumerate(triples):
        subject = row["subject"]
        obj = row["object"]
        relation = row["relation"]
        graph.add_node(subject, type="entity")
        graph.add_node(obj, type="entity")
        graph.add_edge(
            subject,
            obj,
            key=f"e{idx}",
            relation=relation,
            source_sentence_id=row.get("source_sentence_id"),
            confidence=row.get("confidence", 0.0),
        )
    return graph


def save_graph_bundle(graph: nx.MultiDiGraph, graph_index_dir: Path) -> None:
    ensure_dir(graph_index_dir)
    with (graph_index_dir / "graph.pkl").open("wb") as f:
        pickle.dump(graph, f)

    summary = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "entities": sorted(graph.nodes()),
    }
    with (graph_index_dir / "graph_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def load_graph_bundle(graph_index_dir: Path) -> nx.MultiDiGraph:
    graph_path = graph_index_dir / "graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph bundle at {graph_path}")
    with graph_path.open("rb") as f:
        return pickle.load(f)


def find_entity_fuzzy(entity: str, candidates: Sequence[str]) -> Optional[str]:
    value = entity.strip().lower()
    if not value:
        return None
    by_exact = {c.lower(): c for c in candidates}
    if value in by_exact:
        return by_exact[value]

    for c in candidates:
        lc = c.lower()
        if value in lc or lc in value:
            return c
    return None


def extract_main_entity(
    client: OpenAI,
    question: str,
    candidates: Sequence[str],
    model_lite: str,
    model_strong: str,
) -> Tuple[str, str, UsageStats]:
    usage_total = UsageStats()
    system_prompt = (
        "Extract the single most central company or person entity from a question. "
        "Return JSON only: {\"entity\": \"...\"}."
    )
    user_prompt = f"Question: {question}"

    data, usage_lite, _ = chat_json(client, model_lite, system_prompt, user_prompt, max_tokens=150)
    usage_total.add(usage_lite.prompt_tokens, usage_lite.completion_tokens, usage_lite.total_tokens)
    entity = ""
    if data:
        entity = str(data.get("entity", "")).strip()
    matched = find_entity_fuzzy(entity, candidates) if entity else None
    if matched:
        return matched, model_lite, usage_total

    data2, usage_strong, _ = chat_json(client, model_strong, system_prompt, user_prompt, max_tokens=150)
    usage_total.add(
        usage_strong.prompt_tokens, usage_strong.completion_tokens, usage_strong.total_tokens
    )
    entity2 = ""
    if data2:
        entity2 = str(data2.get("entity", "")).strip()
    matched2 = find_entity_fuzzy(entity2, candidates) if entity2 else None
    if matched2:
        return matched2, model_strong, usage_total

    q_lower = question.lower()
    for candidate in candidates:
        if candidate.lower() in q_lower:
            return candidate, "fallback_substring", usage_total

    return "", "not_found", usage_total


def two_hop_evidence(graph: nx.MultiDiGraph, seed: str, max_edges: int = 80) -> List[dict]:
    if seed not in graph:
        return []
    undirected = graph.to_undirected()
    lengths = nx.single_source_shortest_path_length(undirected, seed, cutoff=2)
    selected = set(lengths.keys())

    evidence = []
    for u, v, data in graph.edges(data=True):
        if u in selected and v in selected:
            evidence.append(
                {
                    "subject": u,
                    "relation": data.get("relation", "RELATED_TO"),
                    "object": v,
                    "confidence": data.get("confidence", 0.0),
                }
            )
        if len(evidence) >= max_edges:
            break
    return evidence


def textualize_evidence(evidence: Sequence[dict]) -> str:
    lines = []
    for row in evidence:
        lines.append(f"{row['subject']} --{row['relation']}--> {row['object']}")
    return "\n".join(lines)


def keyword_score(answer: str, expected_keywords: str) -> Tuple[int, int, bool]:
    keywords = [k.strip() for k in expected_keywords.split(";") if k.strip()]
    ans = answer.lower()
    hit = 0
    for key in keywords:
        if key.lower() in ans:
            hit += 1
    total = len(keywords)
    is_correct = total > 0 and hit == total
    return hit, total, is_correct
