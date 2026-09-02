from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, runtime_checkable

from .embedding_vectors import Embedding, normalize_embedding, normalize_embedding_batch


ProviderKind = Literal["local", "cloud"]
DocumentEmbedder = Callable[[Sequence[str]], Sequence[Embedding]]
QueryEmbedder = Callable[[str], Embedding]


@dataclass(frozen=True)
class EmbeddingProviderDescriptor:
    """Stable identity and privacy boundary for an embedding provider."""

    kind: ProviderKind
    model_id: str
    revision: str
    model_fingerprint: str
    dimension: int
    normalization: str = "l2"

    def __post_init__(self) -> None:
        if self.kind not in {"local", "cloud"}:
            raise ValueError("embedding provider kind must be 'local' or 'cloud'")
        for field_name in ("model_id", "revision", "model_fingerprint"):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"embedding provider {field_name} cannot be empty")
        if int(self.dimension) <= 0:
            raise ValueError("embedding provider dimension must be positive")
        if self.normalization != "l2":
            raise ValueError("only l2-normalized embedding indexes are supported")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def identity(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Model execution contract; it does not imply product activation or consent."""

    @property
    def descriptor(self) -> EmbeddingProviderDescriptor:
        ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        ...

    def embed_query(self, text: str) -> Embedding:
        ...


class CallableEmbeddingProvider:
    """Validated adapter for an explicitly supplied local or cloud executor.

    The adapter intentionally does not download a model, read credentials, or
    perform network calls. Product settings and consent remain separate work.
    """

    def __init__(
        self,
        descriptor: EmbeddingProviderDescriptor,
        embed_documents: DocumentEmbedder,
        embed_query: QueryEmbedder,
    ) -> None:
        self._descriptor = descriptor
        self._document_embedder = embed_documents
        self._query_embedder = embed_query

    @property
    def descriptor(self) -> EmbeddingProviderDescriptor:
        return self._descriptor

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return []
        raw_vectors = self._document_embedder(clean_texts)
        return normalize_embedding_batch(
            raw_vectors,
            expected_count=len(clean_texts),
            expected_dim=self.descriptor.dimension,
        )

    def embed_query(self, text: str) -> tuple[float, ...]:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("embedding query cannot be empty")
        return normalize_embedding(
            self._query_embedder(clean_text),
            expected_dim=self.descriptor.dimension,
        )


class LocalEmbeddingProvider(CallableEmbeddingProvider):
    """Explicit local-executor boundary without a bundled model or download."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        model_fingerprint: str,
        dimension: int,
        embed_documents: DocumentEmbedder,
        embed_query: QueryEmbedder,
    ) -> None:
        super().__init__(
            EmbeddingProviderDescriptor(
                kind="local",
                model_id=model_id,
                revision=revision,
                model_fingerprint=model_fingerprint,
                dimension=dimension,
            ),
            embed_documents,
            embed_query,
        )
