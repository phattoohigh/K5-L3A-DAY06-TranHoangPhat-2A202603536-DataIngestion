"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """TODO: implement a provider adapter; do not silently fall back to mock vectors."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        try:
            self.model = SentenceTransformer(model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load embedding model: {model_name}"
            ) from error

        dimensions = self.model.get_embedding_dimension()

        if dimensions is None or dimensions <= 0:
            raise RuntimeError(
                f"Cannot determine dimensions for model: {model_name}"
            )

        self.dimensions = int(dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All document inputs must be strings")

        if any(not text.strip() for text in texts):
            raise ValueError("Document inputs cannot be empty")

        try:
            vectors = self.model.encode_document(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to create document embeddings"
            ) from error

        result = vectors.tolist()

        if len(result) != len(texts):
            raise RuntimeError(
                "Embedding count does not match document count"
            )

        if any(
            len(vector) != self.dimensions
            for vector in result
        ):
            raise RuntimeError(
                "Document embedding dimension mismatch"
            )

        return result

    def embed_query(self, question: str) -> list[float]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")

        if not question.strip():
            raise ValueError("question cannot be empty")

        try:
            vector = self.model.encode_query(
                question,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to create query embedding"
            ) from error

        result = vector.tolist()

        if len(result) != self.dimensions:
            raise RuntimeError(
                "Query embedding dimension mismatch"
            )

        return result