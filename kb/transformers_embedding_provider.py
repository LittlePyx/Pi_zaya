from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .embedding_provider import EmbeddingProviderDescriptor
from .embedding_vectors import Embedding


def pool_hidden_states(
    torch: Any, hidden: Any, attention_mask: Any, pooling: str
) -> Any:
    """Apply one frozen benchmark pooling rule without loading model code."""

    if pooling == "attention_masked_mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if pooling == "cls":
        return hidden[:, 0]
    if pooling != "last_token":
        raise ValueError(f"unsupported pooling rule: {pooling}")
    if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
        return hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, sequence_lengths]


class TransformersPoolingEmbeddingProvider:
    """Optional local Transformers encoder with an immutable pooling contract.

    Product shadowing and offline evaluation share this adapter. Construction is
    cache-only by default; callers must opt into downloads explicitly.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        dimension: int,
        query_prefix: str,
        passage_prefix: str,
        max_length: int,
        batch_size: int,
        device: str = "auto",
        dtype: str = "float32",
        allow_download: bool = False,
        cache_dir: Path | None = None,
        pooling: str = "attention_masked_mean",
        padding_side: str | None = None,
    ) -> None:
        clean_model_id = str(model_id or "").strip()
        clean_revision = str(revision or "").strip()
        if not clean_model_id or not clean_revision:
            raise ValueError("model_id and immutable revision are required")
        if dtype not in {"float32", "float16"}:
            raise ValueError("embedding dtype must be float32 or float16")
        if pooling not in {"attention_masked_mean", "cls", "last_token"}:
            raise ValueError(
                "benchmark pooling must be attention_masked_mean, cls, or last_token"
            )
        if padding_side not in {None, "left", "right"}:
            raise ValueError("benchmark padding_side must be left, right, or omitted")
        self.model_id = clean_model_id
        self.revision = clean_revision
        self.dimension = max(1, int(dimension))
        self.query_prefix = str(query_prefix)
        self.passage_prefix = str(passage_prefix)
        self.max_length = max(8, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.requested_device = str(device or "auto")
        self.dtype = dtype
        self.allow_download = bool(allow_download)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self.pooling = pooling
        self.padding_side = padding_side
        contract = {
            "hf_revision": self.revision,
            "pooling": self.pooling,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_length": self.max_length,
            "dtype": self.dtype,
        }
        # Preserve identities of the already-measured mean-pooling providers.
        # New pooling contracts explicitly include their padding behavior.
        if self.padding_side is not None:
            contract["padding_side"] = self.padding_side
        contract_sha = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._descriptor = EmbeddingProviderDescriptor(
            kind="local",
            model_id=self.model_id,
            revision=self.revision,
            model_fingerprint=f"hf-contract-sha256:{contract_sha}",
            dimension=self.dimension,
        )
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = ""
        self._load_duration_ms = 0.0
        self._document_count = 0
        self._query_count = 0
        self._truncated_document_count = 0
        self._truncated_query_count = 0
        self._max_observed_tokens = 0
        self._document_encode_ms = 0.0
        self._query_encode_ms = 0.0

    @property
    def descriptor(self) -> EmbeddingProviderDescriptor:
        return self._descriptor

    def prepare(self) -> dict[str, Any]:
        self._load()
        return self.runtime_info()

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local Dense retrieval requires optional torch and transformers dependencies"
            ) from exc

        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if self.dtype == "float16" and not device.startswith("cuda"):
            raise RuntimeError("float16 benchmark execution requires CUDA")
        load_kwargs = {
            "revision": self.revision,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "local_files_only": not self.allow_download,
            "trust_remote_code": False,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kwargs)
            if self.padding_side is not None:
                tokenizer.padding_side = self.padding_side
            model = AutoModel.from_pretrained(
                self.model_id,
                use_safetensors=True,
                **load_kwargs,
            )
        except (KeyError, OSError, ValueError) as exc:
            mode = (
                "download was allowed"
                if self.allow_download
                else "cache-only mode is active"
            )
            raise RuntimeError(
                f"could not load pinned model {self.model_id}@{self.revision}; {mode}"
            ) from exc
        hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
        if hidden_size != self.dimension:
            raise RuntimeError(
                f"model hidden size is {hidden_size}, expected {self.dimension}"
            )
        if self.dtype == "float16":
            model = model.half()
        model = model.to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._load_duration_ms = (time.perf_counter() - started) * 1000.0

    def _encode(
        self, texts: Sequence[str], *, prefix: str, kind: str
    ) -> list[tuple[float, ...]]:
        clean_texts = [prefix + str(text or "") for text in texts]
        if not clean_texts:
            return []
        self._load()
        torch = self._torch
        vectors: list[tuple[float, ...]] = []
        started = time.perf_counter()
        truncated = 0
        max_observed = 0
        for offset in range(0, len(clean_texts), self.batch_size):
            batch_texts = clean_texts[offset : offset + self.batch_size]
            raw_tokens = self._tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                verbose=False,
            )["input_ids"]
            raw_lengths = [len(row) for row in raw_tokens]
            max_observed = max(max_observed, max(raw_lengths, default=0))
            truncated += sum(1 for length in raw_lengths if length > self.max_length)
            encoded = self._tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self._model(**encoded)
                hidden = output.last_hidden_state.float()
                attention_mask = encoded["attention_mask"]
                pooled = pool_hidden_states(torch, hidden, attention_mask, self.pooling)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(
                tuple(float(value) for value in row)
                for row in normalized.detach().cpu().tolist()
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._max_observed_tokens = max(self._max_observed_tokens, max_observed)
        if kind == "document":
            self._document_count += len(clean_texts)
            self._truncated_document_count += truncated
            self._document_encode_ms += elapsed_ms
        else:
            self._query_count += len(clean_texts)
            self._truncated_query_count += truncated
            self._query_encode_ms += elapsed_ms
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Embedding]:
        return self._encode(texts, prefix=self.passage_prefix, kind="document")

    def embed_query(self, text: str) -> Embedding:
        vectors = self._encode(
            [str(text or "")], prefix=self.query_prefix, kind="query"
        )
        return vectors[0]

    def content_token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Expose exact fast-tokenizer offsets for benchmark-only child slicing."""

        self._load()
        if not bool(getattr(self._tokenizer, "is_fast", False)):
            raise RuntimeError("focused child slicing requires a fast tokenizer")
        encoded = self._tokenizer(
            str(text or ""),
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        raw_offsets = encoded.get("offset_mapping") or []
        return [(int(start), int(end)) for start, end in raw_offsets]

    def runtime_info(self) -> dict[str, Any]:
        torch_version = (
            str(getattr(self._torch, "__version__", "")) if self._torch else ""
        )
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "provider_identity": self.descriptor.identity,
            "model_fingerprint": self.descriptor.model_fingerprint,
            "dimension": self.dimension,
            "pooling": self.pooling,
            "padding_side": self.padding_side or "tokenizer_default",
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self._device or self.requested_device,
            "dtype": self.dtype,
            "allow_download": self.allow_download,
            "trust_remote_code": False,
            "load_duration_ms": round(self._load_duration_ms, 3),
            "document_count": self._document_count,
            "query_count": self._query_count,
            "truncated_document_count": self._truncated_document_count,
            "truncated_query_count": self._truncated_query_count,
            "max_observed_tokens": self._max_observed_tokens,
            "document_encode_ms": round(self._document_encode_ms, 3),
            "query_encode_ms": round(self._query_encode_ms, 3),
            "torch_version": torch_version,
        }


# Backwards-compatible name for the already-frozen E5 benchmark contracts.
TransformersMeanPoolEmbeddingProvider = TransformersPoolingEmbeddingProvider
