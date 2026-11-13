"""Implement fixed-size and structure-aware chunking."""
import hashlib
import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """TODO: split text into overlapping windows and retain page metadata."""
    if size <= 0:
        raise ValueError("size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = page.text.strip()

    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + size, len(text))

        # Keep the fixed-size baseline while avoiding cuts inside a word.
        if end < len(text):
            boundary = text.rfind(" ", start + max(1, size // 2), end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            id_payload = (
                f"{page.document_id}|"
                f"{page.page}|"
                f"fixed|"
                f"{chunk_index}|"
                f"{chunk_text}"
            )

            digest = hashlib.sha256(
                id_payload.encode("utf-8")
            ).hexdigest()[:16]

            chunk_id = (
                f"{page.document_id}-"
                f"p{page.page}-"
                f"fixed-"
                f"{chunk_index:04d}-"
                f"{digest}"
            )

            metadata = dict(page.metadata)
            metadata.update(
                {
                    "chunking": "fixed_size",
                    "chunk_size": size,
                    "overlap": overlap,
                    "char_start": start,
                    "char_end": end,
                    "char_count": len(chunk_text),
                }
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

            chunk_index += 1

        if end >= len(text):
            break

        start = max(0, end - overlap)
        if start > 0:
            next_boundary = text.find(" ", start)
            if 0 <= next_boundary < end:
                start = next_boundary + 1

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """TODO: group paragraphs, split oversized paragraphs and retain metadata."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    text = page.text.strip()

    if not text:
        return []

    # PDF extraction often inserts visual line breaks that are not semantic
    # paragraph boundaries. Rejoin those lines, then split at sentence ends.
    normalized = re.sub(r"\s+", " ", text).strip()
    paragraphs = [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+(?=(?:[A-Z0-9\"“]))",
            normalized,
        )
        if part.strip()
    ]

    pieces: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_size:
            pieces.append(paragraph)
            continue

        words = paragraph.split()
        current_words: list[str] = []
        current_length = 0

        for word in words:
            # Handle a token that is longer than max_size.
            if len(word) > max_size:
                if current_words:
                    pieces.append(" ".join(current_words))
                    current_words = []
                    current_length = 0

                for start in range(0, len(word), max_size):
                    pieces.append(word[start:start + max_size])

                continue

            proposed_length = (
                current_length + len(word)
                if not current_words
                else current_length + 1 + len(word)
            )

            if proposed_length <= max_size:
                current_words.append(word)
                current_length = proposed_length
            else:
                pieces.append(" ".join(current_words))
                current_words = [word]
                current_length = len(word)

        if current_words:
            pieces.append(" ".join(current_words))

    grouped_texts: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for piece in pieces:
        separator_length = 2 if current_parts else 0
        proposed_length = (
            current_length
            + separator_length
            + len(piece)
        )

        if proposed_length <= max_size:
            current_parts.append(piece)
            current_length = proposed_length
        else:
            if current_parts:
                grouped_texts.append("\n\n".join(current_parts))

            current_parts = [piece]
            current_length = len(piece)

    if current_parts:
        grouped_texts.append("\n\n".join(current_parts))

    # Avoid leaving a very small trailing fragment.
    minimum_chunk_size = min(180, max_size)

    if (
        len(grouped_texts) >= 2
        and len(grouped_texts[-1])
        < minimum_chunk_size
    ):
        previous_text = grouped_texts[-2]
        trailing_text = grouped_texts[-1]

        combined_text = (
            previous_text
            + "\n\n"
            + trailing_text
        )

        if len(combined_text) <= max_size:
            grouped_texts[-2] = combined_text
            grouped_texts.pop()
        else:
            previous_words = previous_text.split()
            moved_words: list[str] = []

            while (
                len(trailing_text)
                < minimum_chunk_size
                and len(previous_words) > 1
            ):
                moved_words.insert(
                    0,
                    previous_words.pop(),
                )

                trailing_text = (
                    " ".join(moved_words)
                    + "\n\n"
                    + grouped_texts[-1]
                )

            rebalanced_previous = " ".join(
                previous_words
            )

            if rebalanced_previous:
                grouped_texts[-2] = (
                    rebalanced_previous
                )
                grouped_texts[-1] = trailing_text

    chunks: list[Chunk] = []

    for chunk_index, chunk_text in enumerate(grouped_texts):
        id_payload = (
            f"{page.document_id}|"
            f"{page.page}|"
            f"paragraph|"
            f"{chunk_index}|"
            f"{chunk_text}"
        )

        digest = hashlib.sha256(
            id_payload.encode("utf-8")
        ).hexdigest()[:16]

        chunk_id = (
            f"{page.document_id}-"
            f"p{page.page}-"
            f"paragraph-"
            f"{chunk_index:04d}-"
            f"{digest}"
        )

        metadata = dict(page.metadata)
        metadata.update(
            {
                "chunking": "by_paragraph",
                "max_size": max_size,
                "overlap": 0,
                "char_count": len(chunk_text),
            }
        )

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=metadata,
            )
        )

    return chunks
