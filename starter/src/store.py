"""TODO: store and search vectors using PostgreSQL + pgvector."""
from collections import defaultdict
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """TODO: retain connection settings; validate model dimensions."""
        if not database_url.strip():
            raise ValueError("database_url cannot be empty")

        if dimensions <= 0:
            raise ValueError("dimensions must be greater than 0")

        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """TODO: create table with vector column and source/page/filter metadata."""
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                "CREATE EXTENSION IF NOT EXISTS vector"
            )

            register_vector(connection)

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR({self.dimensions}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    document_chunks_strategy_index
                ON document_chunks (strategy)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    document_chunks_document_index
                ON document_chunks (strategy, document_id)
                """
            )

            row = connection.execute(
                """
                SELECT format_type(
                    attribute.atttypid,
                    attribute.atttypmod
                )
                FROM pg_attribute AS attribute
                JOIN pg_class AS table_info
                  ON table_info.oid = attribute.attrelid
                WHERE table_info.relname = 'document_chunks'
                  AND attribute.attname = 'embedding'
                  AND attribute.attnum > 0
                  AND NOT attribute.attisdropped
                """
            ).fetchone()

            expected_type = f"vector({self.dimensions})"

            if row is None or row[0] != expected_type:
                actual_type = None if row is None else row[0]

                raise RuntimeError(
                    "Existing vector column dimension does not match "
                    f"the model: expected={expected_type}, "
                    f"actual={actual_type}"
                )

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """TODO: validate lengths/dimensions and write stable IDs without duplicates.

        When re-ingesting changed source documents, remove obsolete chunks too.
        """
        if not strategy.strip():
            raise ValueError("strategy cannot be empty")

        if len(chunks) != len(vectors):
            raise ValueError(
                "chunks and vectors must have the same length"
            )

        if not chunks:
            return

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Duplicate chunk IDs found in the input"
            )

        for index, vector in enumerate(vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    "Vector dimension mismatch at "
                    f"index={index}: "
                    f"expected={self.dimensions}, "
                    f"actual={len(vector)}"
                )

        chunks_by_document: dict[str, list[str]] = defaultdict(list)

        for chunk in chunks:
            chunks_by_document[chunk.document_id].append(
                chunk.chunk_id
            )

        rows = []

        for chunk, vector in zip(chunks, vectors):
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    Vector(vector),
                )
            )

        with psycopg.connect(self.database_url) as connection:
            register_vector(connection)

            with connection.cursor() as cursor:
                for document_id, current_ids in (
                    chunks_by_document.items()
                ):
                    cursor.execute(
                        """
                        DELETE FROM document_chunks
                        WHERE strategy = %s
                          AND document_id = %s
                          AND NOT (
                              chunk_id = ANY(%s)
                          )
                        """,
                        (
                            strategy,
                            document_id,
                            current_ids,
                        ),
                    )

                cursor.executemany(
                    """
                    INSERT INTO document_chunks (
                        strategy,
                        chunk_id,
                        document_id,
                        source,
                        page,
                        content,
                        metadata,
                        embedding
                    )
                    VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (strategy, chunk_id)
                    DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    rows,
                )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """TODO: filter permitted metadata in SQL, rank by cosine distance, return top-k.

        Return chunk ID, text, score and metadata. Parameterize user-supplied values.
        """
        if not strategy.strip():
            raise ValueError("strategy cannot be empty")

        if len(query_vector) != self.dimensions:
            raise ValueError(
                "Query vector dimension mismatch: "
                f"expected={self.dimensions}, "
                f"actual={len(query_vector)}"
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        allowed_filters = {
            "country",
            "year",
            "topic",
            "document_type",
            "language",
        }

        filters = filters or {}

        unsupported_filters = (
            set(filters) - allowed_filters
        )

        if unsupported_filters:
            raise ValueError(
                "Unsupported metadata filters: "
                f"{sorted(unsupported_filters)}"
            )

        where_clauses = ["strategy = %s"]
        where_parameters: list[Any] = [strategy]

        for key, value in filters.items():
            if value is None:
                raise ValueError(
                    f"Filter value cannot be null: {key}"
                )

            where_clauses.append(
                "metadata ->> %s = %s"
            )

            where_parameters.extend(
                [key, str(value)]
            )

        where_sql = " AND ".join(where_clauses)
        query_embedding = Vector(query_vector)

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source,
                page,
                content,
                metadata,
                1 - (embedding <=> %s) AS score
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
        """

        parameters = [
            query_embedding,
            *where_parameters,
            query_embedding,
            top_k,
        ]

        with psycopg.connect(self.database_url) as connection:
            register_vector(connection)

            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        results: list[dict[str, Any]] = []

        for row in rows:
            results.append(
                {
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "source": row[2],
                    "page": row[3],
                    "text": row[4],
                    "metadata": row[5],
                    "score": float(row[6]),
                }
            )

        return results