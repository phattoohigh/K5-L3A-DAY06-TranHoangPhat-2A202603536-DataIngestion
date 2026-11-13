"""Suggested pipeline entry point; TODO: implement the orchestration."""
import json
import os
import statistics
import time
import unicodedata
import re
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf, remove_repeated_marginal_lines
from .models import Chunk, DocumentPage
from .store import PgVectorStore


def parse_page_ranges(ranges: list[str]) -> set[int]:
    """Convert manifest page ranges into physical page numbers."""
    pages: set[int] = set()

    for value in ranges:
        if "-" in value:
            start_text, end_text = value.split(
                "-",
                maxsplit=1,
            )

            start = int(start_text)
            end = int(end_text)

            if start <= 0 or end < start:
                raise ValueError(
                    f"Invalid page range: {value}"
                )

            pages.update(range(start, end + 1))
        else:
            page = int(value)

            if page <= 0:
                raise ValueError(
                    f"Invalid page number: {value}"
                )

            pages.add(page)

    return pages


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json(
    path: Path,
    data: Any,
) -> None:
    """Write a JSON artifact using UTF-8."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def normalize_for_validation(text: str) -> str:
    """Normalize evidence before corpus validation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


_LEXICAL_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "did", "do", "for",
    "from", "how", "in", "into", "is", "it", "of", "on", "or", "the",
    "their", "to", "was", "were", "what", "which", "with",
}


def lexical_tokens(text: str) -> set[str]:
    """Return conservative English word/number tokens for hybrid reranking."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)*|\d+(?:\.\d+)?", normalized)
    return {
        token
        for token in tokens
        if token not in _LEXICAL_STOPWORDS and len(token) > 1
    }


def hybrid_rerank(
    query: str,
    hits: list[dict[str, Any]],
    dense_weight: float,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rerank dense candidates with an IDF-weighted lexical overlap score."""
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be between 0 and 1")
    if not hits:
        return []

    query_tokens = lexical_tokens(query)
    document_tokens = [lexical_tokens(str(hit.get("text", ""))) for hit in hits]
    document_frequency = {
        token: sum(token in tokens for tokens in document_tokens)
        for token in query_tokens
    }
    token_weights = {
        token: math.log((len(hits) + 1) / (count + 1)) + 1
        for token, count in document_frequency.items()
    }
    maximum_lexical = sum(token_weights.values()) or 1.0
    dense_scores = [float(hit.get("score", 0.0)) for hit in hits]
    dense_min = min(dense_scores)
    dense_max = max(dense_scores)
    dense_span = dense_max - dense_min

    reranked: list[dict[str, Any]] = []
    for hit, tokens, dense_score in zip(hits, document_tokens, dense_scores):
        lexical_score = sum(
            weight for token, weight in token_weights.items() if token in tokens
        ) / maximum_lexical
        normalized_dense = (
            (dense_score - dense_min) / dense_span if dense_span > 0 else 1.0
        )
        hybrid_score = (
            dense_weight * normalized_dense
            + (1 - dense_weight) * lexical_score
        )
        item = dict(hit)
        item["dense_score"] = dense_score
        item["lexical_score"] = lexical_score
        item["hybrid_score"] = hybrid_score
        item["score"] = hybrid_score
        reranked.append(item)

    reranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return reranked[:top_k]


def load_corpus(
    repo_root: Path,
    manifest: dict[str, Any],
) -> list[DocumentPage]:
    """Load every manifest document and validate page counts."""
    documents = manifest.get("documents", [])

    if not documents:
        raise ValueError(
            "Manifest does not contain documents"
        )

    all_pages: list[DocumentPage] = []

    for document in documents:
        included_pages = parse_page_ranges(
            document["included_page_ranges"]
        )

        pdf_path = (
            repo_root
            / document["source_path_or_url"]
        )

        metadata = {
            "country": document["country"],
            "year": document["year"],
            "topic": document["topic"],
            "document_type": document[
                "document_type"
            ],
            "language": document["language"],
        }

        pages = load_pdf(
            path=pdf_path,
            document_id=document["source_id"],
            source=document["source_path_or_url"],
            metadata=metadata,
            included_pages=included_pages,
        )

        expected_count = document[
            "included_content_page_count"
        ]

        if len(pages) != expected_count:
            raise RuntimeError(
                "Loaded page count mismatch for "
                f"{document['source_id']}: "
                f"expected={expected_count}, "
                f"actual={len(pages)}"
            )

        print(
            f"Loaded {document['source_id']}: "
            f"{len(pages)} pages"
        )

        all_pages.extend(pages)

    expected_total = manifest[
        "total_included_content_pages"
    ]

    if len(all_pages) != expected_total:
        raise RuntimeError(
            "Total corpus page count mismatch: "
            f"expected={expected_total}, "
            f"actual={len(all_pages)}"
        )

    if len(all_pages) < 30:
        raise RuntimeError(
            "Corpus must contain at least "
            "30 valid content pages"
        )

    cleaned_pages = remove_repeated_marginal_lines(all_pages)

    if len(cleaned_pages) != len(all_pages):
        raise RuntimeError(
            "Header/footer cleaning unexpectedly removed whole pages"
        )

    return cleaned_pages


def validate_benchmark(
    questions: list[dict[str, Any]],
    pages: list[DocumentPage],
) -> None:
    """Validate splits, filters and gold evidence."""
    if len(questions) < 20:
        raise ValueError(
            "Benchmark must contain at least 20 questions"
        )

    ids = [
        question["id"]
        for question in questions
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Benchmark contains duplicate question IDs"
        )

    dev_count = sum(
        question["split"] == "dev"
        for question in questions
    )

    test_count = sum(
        question["split"] == "test"
        for question in questions
    )

    if dev_count < 15:
        raise ValueError(
            "Benchmark requires at least 15 dev questions"
        )

    if test_count < 5:
        raise ValueError(
            "Benchmark requires at least 5 test questions"
        )

    filter_questions = [
        question
        for question in questions
        if question.get("filter")
    ]

    if len(filter_questions) < 3:
        raise ValueError(
            "Benchmark requires at least "
            "3 metadata-filter questions"
        )

    test_filter_count = sum(
        question["split"] == "test"
        for question in filter_questions
    )

    if test_filter_count < 1:
        raise ValueError(
            "At least one test question "
            "must use a metadata filter"
        )

    page_lookup = {
        (
            page.document_id,
            page.page,
        ): page
        for page in pages
    }

    for question in questions:
        source_id = question["gold_source_id"]

        gold_pages = [
            int(page)
            for page in question.get(
                "gold_pages",
                [question["gold_page"]],
            )
        ]

        evidence_quotes = question.get(
            "evidence_quotes",
            [question["evidence_quote"]],
        )

        available_pages = []

        for gold_page in gold_pages:
            page = page_lookup.get(
                (source_id, gold_page)
            )

            if page is not None:
                available_pages.append(page)

        if len(available_pages) != len(gold_pages):
            raise ValueError(
                f"One or more gold pages were not found "
                f"for {question['id']}: "
                f"{source_id}, pages={gold_pages}"
            )

        evidence_found = False

        for page in available_pages:
            normalized_page = (
                normalize_for_validation(
                    page.text
                )
            )

            for evidence_quote in evidence_quotes:
                normalized_evidence = (
                    normalize_for_validation(
                        evidence_quote
                    )
                )

                if (
                    normalized_evidence
                    in normalized_page
                ):
                    evidence_found = True
                    break

            if evidence_found:
                break

        if not evidence_found:
            raise ValueError(
                f"Evidence not found for "
                f"{question['id']} in "
                f"{source_id}, pages={gold_pages}"
            )


def build_chunks(
    pages: list[DocumentPage],
    fixed_chunk_size: int = 800,
    fixed_overlap: int = 100,
    structured_chunk_size: int = 800,
) -> dict[str, list[Chunk]]:
    """Build controlled A, B and C strategies."""
    strategy_a: list[Chunk] = []
    strategy_b: list[Chunk] = []

    for page in pages:
        strategy_a.extend(
            fixed_size(
                page=page,
                size=fixed_chunk_size,
                overlap=fixed_overlap,
            )
        )

        strategy_b.extend(
            by_paragraph(
                page=page,
                max_size=structured_chunk_size,
            )
        )

    # Strategy C deliberately uses the same chunks as B.
    strategy_c = strategy_b

    if not strategy_a or not strategy_b:
        raise RuntimeError(
            "Chunking produced no chunks"
        )

    strategy_b_ids = [
        chunk.chunk_id
        for chunk in strategy_b
    ]

    strategy_c_ids = [
        chunk.chunk_id
        for chunk in strategy_c
    ]

    if strategy_b_ids != strategy_c_ids:
        raise RuntimeError(
            "Strategies B and C must use identical chunks"
        )

    all_ids = [
        f"A:{chunk.chunk_id}"
        for chunk in strategy_a
    ] + [
        f"B:{chunk.chunk_id}"
        for chunk in strategy_b
    ]

    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(
            "Duplicate chunk IDs detected"
        )

    return {
        "A": strategy_a,
        "B": strategy_b,
        "C": strategy_c,
    }


def chunk_statistics(
    chunks: list[Chunk],
) -> dict[str, Any]:
    """Calculate chunk length statistics."""
    lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    return {
        "chunk_count": len(chunks),
        "minimum_characters": min(lengths),
        "maximum_characters": max(lengths),
        "mean_characters": statistics.mean(lengths),
        "median_characters": statistics.median(lengths),
        "total_characters": sum(lengths),
    }


def main() -> None:
    """TODO:

    1. Load the corpus and benchmark; validate at least 30 content pages.
    2. Build strategies A (fixed), B (paragraph) and C (B + metadata filter).
    3. Embed with the same model and store vectors/metadata per strategy.
    4. Run the same dev/test questions and save full top-3 traces.
    5. Report retrieval metrics and prepare samples for manual assessment.

    Follow the five dimensions in ../../BENCHMARK.md.
    """
    started_at = time.perf_counter()

    starter_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]

    manifest_path = (
        repo_root
        / "data"
        / "manifest.json"
    )

    benchmark_path = (
        repo_root
        / "benchmark"
        / "questions.json"
    )

    configured_results = os.getenv("RESULTS_DIRECTORY", "results")
    results_directory = starter_root / configured_results

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_dotenv(starter_root / ".env")

    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing from starter/.env"
        )

    model_name = (
        os.getenv("EMBEDDING_MODEL")
        or (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

    manifest = load_json(manifest_path)
    benchmark = load_json(benchmark_path)

    questions = benchmark.get(
        "questions",
        [],
    )

    requested_split = os.getenv("BENCHMARK_SPLIT", "all").strip().lower()
    if requested_split not in {"all", "dev", "test"}:
        raise ValueError("BENCHMARK_SPLIT must be all, dev, or test")

    all_questions = questions
    if requested_split != "all":
        questions = [
            question
            for question in all_questions
            if question["split"] == requested_split
        ]

    hybrid_candidates = int(os.getenv("HYBRID_CANDIDATES", "20"))
    hybrid_dense_weight = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.35"))
    use_hybrid = os.getenv("USE_HYBRID", "1").strip() not in {"0", "false", "no"}

    print("Loading corpus...")

    ingestion_started = time.perf_counter()

    pages = load_corpus(
        repo_root=repo_root,
        manifest=manifest,
    )

    ingestion_seconds = (
        time.perf_counter()
        - ingestion_started
    )

    print(
        f"Loaded {len(pages)} total pages "
        f"in {ingestion_seconds:.2f} seconds"
    )

    print("Validating benchmark...")

    validate_benchmark(
        questions=all_questions,
        pages=pages,
    )

    print(
        f"Validated {len(all_questions)} questions; "
        f"running {len(questions)} for split={requested_split}"
    )

    print("Building chunks...")

    chunking_started = time.perf_counter()

    fixed_chunk_size = int(os.getenv("FIXED_CHUNK_SIZE", "800"))
    fixed_overlap = int(os.getenv("FIXED_OVERLAP", "100"))
    structured_chunk_size = int(
        os.getenv("STRUCTURED_CHUNK_SIZE", "650")
    )

    strategies = build_chunks(
        pages,
        fixed_chunk_size=fixed_chunk_size,
        fixed_overlap=fixed_overlap,
        structured_chunk_size=structured_chunk_size,
    )

    chunking_seconds = (
        time.perf_counter()
        - chunking_started
    )

    for strategy, chunks in strategies.items():
        print(
            f"Strategy {strategy}: "
            f"{len(chunks)} chunks"
        )

    print("Loading embedding model...")

    embedder = MyEmbedder(model_name)

    print(
        f"Model: {embedder.model_name}"
    )

    print(
        f"Dimensions: {embedder.dimensions}"
    )

    store = PgVectorStore(
        database_url=database_url,
        dimensions=embedder.dimensions,
    )

    store.create_schema()

    embedding_seconds: dict[str, float] = {}
    vectors_by_strategy: dict[
        str,
        list[list[float]],
    ] = {}

    for strategy in ("A", "B"):
        chunks = strategies[strategy]

        print(
            f"Embedding strategy {strategy}..."
        )

        embedding_started = time.perf_counter()

        vectors = embedder.embed_documents(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        embedding_seconds[strategy] = (
            time.perf_counter()
            - embedding_started
        )

        vectors_by_strategy[strategy] = vectors

    # C uses exactly the same chunks and vectors as B.
    vectors_by_strategy["C"] = (
        vectors_by_strategy["B"]
    )

    embedding_seconds["C"] = 0.0

    ingest_store_seconds: dict[str, float] = {}

    for strategy in ("A", "B", "C"):
        print(
            f"Upserting strategy {strategy}..."
        )

        store_started = time.perf_counter()

        store.upsert(
            strategy=strategy,
            chunks=strategies[strategy],
            vectors=vectors_by_strategy[strategy],
        )

        ingest_store_seconds[strategy] = (
            time.perf_counter()
            - store_started
        )

    all_traces: list[dict[str, Any]] = []
    query_seconds: dict[str, float] = {}

    for strategy in ("A", "B", "C"):
        print(
            f"Running benchmark for strategy "
            f"{strategy}..."
        )

        strategy_started = time.perf_counter()
        strategy_traces = []

        for question in questions:
            retrieval_query = question.get(
                "retrieval_query",
                question["question"],
            )
            query_vector = embedder.embed_query(retrieval_query)

            # Only strategy C applies question metadata filters.
            filters = (
                question.get("filter")
                if strategy == "C"
                else None
            )

            hits = store.search(
                strategy=strategy,
                query_vector=query_vector,
                top_k=hybrid_candidates if use_hybrid else 3,
                filters=filters,
            )

            if use_hybrid:
                hits = hybrid_rerank(
                    retrieval_query,
                    hits,
                    dense_weight=hybrid_dense_weight,
                    top_k=3,
                )

            evaluation_question = dict(question)
            evaluation_question["strategy"] = strategy

            trace = evaluate_query(
                question=evaluation_question,
                hits=hits,
            )
            trace["retrieval_query"] = retrieval_query

            strategy_traces.append(trace)
            all_traces.append(trace)

        query_seconds[strategy] = (
            time.perf_counter()
            - strategy_started
        )

        write_json(
            results_directory
            / f"traces_{strategy}.json",
            strategy_traces,
        )

    metrics = summarize(all_traces)

    metrics.update(
        {
            "model_name": embedder.model_name,
            "dimensions": embedder.dimensions,
            "similarity_metric": "cosine",
            "search_mode": "exact",
            "retrieval_mode": "hybrid_dense_lexical" if use_hybrid else "dense",
            "hybrid_candidates": hybrid_candidates if use_hybrid else None,
            "hybrid_dense_weight": hybrid_dense_weight if use_hybrid else None,
            "cost": 0,
            "benchmark_split": requested_split,
            "selection_rule": (
                "Prioritize dev Evidence Hit@3, "
                "then manual grounding, Hit@1, "
                "coherence, and resource use."
            ),
            "strategy_parameters": {
                "A": {
                    "chunk_size": fixed_chunk_size,
                    "overlap": fixed_overlap,
                },
                "B": {
                    "max_size": structured_chunk_size,
                    "filter": False,
                },
                "C": {
                    "max_size": structured_chunk_size,
                    "filter": True,
                },
            },
            "chunk_statistics": {
                strategy: chunk_statistics(chunks)
                for strategy, chunks
                in strategies.items()
            },
            "timings_seconds": {
                "ingestion": ingestion_seconds,
                "chunking": chunking_seconds,
                "embedding": embedding_seconds,
                "vector_store_upsert": (
                    ingest_store_seconds
                ),
                "queries": query_seconds,
                "total": (
                    time.perf_counter()
                    - started_at
                ),
            },
        }
    )

    write_json(
        results_directory
        / "metrics.json",
        metrics,
    )

    write_json(
        results_directory
        / "ingest_summary.json",
        {
            "corpus_id": manifest["corpus_id"],
            "document_count": len(
                manifest["documents"]
            ),
            "loaded_content_pages": len(pages),
            "model_name": embedder.model_name,
            "dimensions": embedder.dimensions,
            "chunk_statistics": {
                strategy: chunk_statistics(chunks)
                for strategy, chunks
                in strategies.items()
            },
        },
    )

    print("\nBenchmark summary:")

    for group in metrics["groups"]:
        print(
            f"Strategy={group['strategy']} "
            f"Split={group['split']} "
            f"Hit@1="
            f"{group['hit_at_1_count']}/"
            f"{group['hit_at_1_denominator']} "
            f"Evidence Hit@3="
            f"{group['evidence_hit_at_3_count']}/"
            f"{group['evidence_hit_at_3_denominator']}"
        )

    print(
        "\nResults written to:",
        results_directory,
    )

    print(
        "Pipeline completed in "
        f"{metrics['timings_seconds']['total']:.2f} "
        "seconds"
    )


if __name__ == "__main__":
    main()
