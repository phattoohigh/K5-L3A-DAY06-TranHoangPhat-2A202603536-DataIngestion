"""TODO: evaluate retrieved evidence and preserve per-query traces."""
import re
import unicodedata
from collections import defaultdict
from typing import Any


def _normalize_for_matching(text: str) -> str:
    """Normalize text for conservative evidence matching."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _as_list(value: Any) -> list[Any]:
    """Convert a scalar or list-like gold value to a list."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """TODO: check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    required_fields = {
        "id",
        "split",
        "question",
        "expected_answer",
        "gold_source_id",
        "gold_page",
        "evidence_quote",
    }

    missing_fields = required_fields - question.keys()

    if missing_fields:
        raise ValueError(
            f"Question is missing fields: {sorted(missing_fields)}"
        )

    if question["split"] not in {"dev", "test"}:
        raise ValueError(
            f"Invalid split for {question['id']}: "
            f"{question['split']}"
        )

    gold_source_ids = _as_list(
        question.get(
            "gold_source_ids",
            question["gold_source_id"],
        )
    )

    gold_pages = _as_list(
        question.get(
            "gold_pages",
            question["gold_page"],
        )
    )

    evidence_quotes = _as_list(
        question.get(
            "evidence_quotes",
            question["evidence_quote"],
        )
    )

    gold_source_ids = {
        str(source_id)
        for source_id in gold_source_ids
    }

    gold_pages = {
        int(page)
        for page in gold_pages
    }

    normalized_evidence = [
        _normalize_for_matching(str(evidence))
        for evidence in evidence_quotes
        if str(evidence).strip()
    ]

    if not gold_source_ids:
        raise ValueError(
            f"Question {question['id']} has no gold source"
        )

    if not gold_pages:
        raise ValueError(
            f"Question {question['id']} has no gold page"
        )

    if not normalized_evidence:
        raise ValueError(
            f"Question {question['id']} has no evidence"
        )

    evaluated_hits: list[dict[str, Any]] = []

    for rank, hit in enumerate(hits[:3], start=1):
        hit_text = str(hit.get("text", ""))
        hit_document_id = str(
            hit.get("document_id", "")
        )
        hit_page = hit.get("page")

        source_match = (
            hit_document_id in gold_source_ids
        )

        try:
            page_match = int(hit_page) in gold_pages
        except (TypeError, ValueError):
            page_match = False

        normalized_hit_text = _normalize_for_matching(
            hit_text
        )

        matching_evidence = [
            evidence
            for evidence in normalized_evidence
            if evidence in normalized_hit_text
        ]

        evidence_match = bool(matching_evidence)

        is_hit = (
            source_match
            and page_match
            and evidence_match
        )

        evaluated_hit = dict(hit)
        evaluated_hit["rank"] = rank
        evaluated_hit["source_match"] = source_match
        evaluated_hit["page_match"] = page_match
        evaluated_hit["evidence_match"] = evidence_match
        evaluated_hit["is_hit"] = is_hit

        evaluated_hits.append(evaluated_hit)

    hit_at_1 = bool(
        evaluated_hits
        and evaluated_hits[0]["is_hit"]
    )

    evidence_hit_at_3 = any(
        hit["is_hit"]
        for hit in evaluated_hits
    )

    return {
        "question_id": question["id"],
        "strategy": question.get(
            "strategy",
            "unknown",
        ),
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_ids": sorted(gold_source_ids),
        "gold_pages": sorted(gold_pages),
        "evidence_quotes": evidence_quotes,
        "filter": question.get("filter"),
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "retrieved_count": len(evaluated_hits),
        "hits": evaluated_hits,
    }


def summarize(
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    """TODO: aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for trace in traces:
        strategy = str(
            trace.get("strategy", "unknown")
        )
        split = str(
            trace.get("split", "unknown")
        )

        grouped[(strategy, split)].append(trace)

    summaries: list[dict[str, Any]] = []

    for strategy, split in sorted(grouped):
        group = grouped[(strategy, split)]
        total = len(group)

        hit_at_1_count = sum(
            bool(trace.get("hit_at_1"))
            for trace in group
        )

        evidence_hit_at_3_count = sum(
            bool(trace.get("evidence_hit_at_3"))
            for trace in group
        )

        summaries.append(
            {
                "strategy": strategy,
                "split": split,
                "question_count": total,
                "hit_at_1_count": hit_at_1_count,
                "hit_at_1_denominator": total,
                "hit_at_1_rate": (
                    hit_at_1_count / total
                    if total
                    else 0.0
                ),
                "evidence_hit_at_3_count": (
                    evidence_hit_at_3_count
                ),
                "evidence_hit_at_3_denominator": total,
                "evidence_hit_at_3_rate": (
                    evidence_hit_at_3_count / total
                    if total
                    else 0.0
                ),
            }
        )

    return {
        "trace_count": len(traces),
        "groups": summaries,
    }