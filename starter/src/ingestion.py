"""Read and normalize source data."""

import re
import unicodedata
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    included_pages: set[int] | None = None,
) -> list[DocumentPage]:
    """Extract selected physical PDF pages and retain source metadata."""
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    pages: list[DocumentPage] = []
    metadata = metadata or {}

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        if included_pages is not None and page_number not in included_pages:
            continue

        try:
            raw_text = pdf_page.extract_text() or ""
        except Exception as error:
            print(
                f"Error extracting document={document_id} "
                f"page={page_number}: {error}"
            )
            continue

        text = normalize_text(raw_text)

        if not text:
            print(
                f"Warning: empty page "
                f"document={document_id} page={page_number}"
            )
            continue

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata=dict(metadata),
            )
        )

    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraphs."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines: list[str] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        normalized_lines.append(line)

    text = "\n".join(normalized_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_repeated_marginal_lines(
    pages: list[DocumentPage],
    edge_line_count: int = 5,
    repeat_ratio: float = 0.15,
    maximum_line_length: int = 160,
) -> list[DocumentPage]:
    """Remove repeated PDF headers and footers within each document."""
    if not 0 < repeat_ratio <= 1:
        raise ValueError("repeat_ratio must be in (0, 1]")

    pages_by_document: dict[str, list[DocumentPage]] = defaultdict(list)
    for page in pages:
        pages_by_document[page.document_id].append(page)

    cleaned_pages: list[DocumentPage] = []

    def fingerprint(line: str) -> str:
        value = unicodedata.normalize("NFKC", line).casefold()
        value = re.sub(r"\d+", "#", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip(" -|·")

    for document_pages in pages_by_document.values():
        edge_counts: Counter[str] = Counter()
        page_edges: list[set[str]] = []

        for page in document_pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            edge_lines = lines[:edge_line_count] + lines[-edge_line_count:]
            fingerprints = {
                fingerprint(line)
                for line in edge_lines
                if 0 < len(line) <= maximum_line_length
            }
            page_edges.append(fingerprints)
            edge_counts.update(fingerprints)

        threshold = max(3, ceil(len(document_pages) * repeat_ratio))
        repeated = {
            value
            for value, count in edge_counts.items()
            if value and count >= threshold
        }

        for page, edge_fingerprints in zip(document_pages, page_edges):
            lines = page.text.splitlines()
            nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
            edge_indexes = set(
                nonempty_indexes[:edge_line_count]
                + nonempty_indexes[-edge_line_count:]
            )
            retained: list[str] = []

            for index, line in enumerate(lines):
                stripped = line.strip()
                line_fingerprint = fingerprint(stripped)
                is_bare_page_number = bool(
                    re.fullmatch(r"(?:page\s+)?\d+", stripped, re.I)
                )
                is_repeated_edge = (
                    index in edge_indexes
                    and line_fingerprint in repeated
                    and line_fingerprint in edge_fingerprints
                )

                if stripped and (is_bare_page_number or is_repeated_edge):
                    continue
                retained.append(line)

            cleaned_text = normalize_text("\n".join(retained))
            if cleaned_text:
                cleaned_pages.append(
                    DocumentPage(
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=cleaned_text,
                        metadata=dict(page.metadata),
                    )
                )

    return cleaned_pages
