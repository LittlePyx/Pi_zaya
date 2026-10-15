from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kb.cross_encoder_reranker import Pair


class TransformersSequenceClassificationReranker:
    """Benchmark-only, cache-first Transformers cross-encoder scorer."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        max_length: int,
        batch_size: int,
        device: str = "auto",
        dtype: str = "float32",
        allow_download: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_id = str(model_id or "").strip()
        self.revision = str(revision or "").strip()
        if not self.model_id or not self.revision:
            raise ValueError("model_id and immutable revision are required")
        if dtype not in {"float32", "float16"}:
            raise ValueError("benchmark dtype must be float32 or float16")
        self.max_length = max(8, int(max_length))
        self.batch_size = max(1, int(batch_size))
        self.requested_device = str(device or "auto")
        self.dtype = dtype
        self.allow_download = bool(allow_download)
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        contract = {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": "transformers_sequence_classification",
            "tokenizer": "sentencepiece_slow",
            "truncation": "only_second",
            "max_length": self.max_length,
            "dtype": self.dtype,
        }
        self.contract_identity = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = ""
        self._load_duration_ms = 0.0
        self._pair_count = 0
        self._truncated_pair_count = 0
        self._max_observed_tokens = 0
        self._score_duration_ms = 0.0

    def prepare(self) -> dict[str, Any]:
        self._load()
        return self.runtime_info()

    def _load(self) -> None:
        if self._model is not None:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local cross-encoder benchmark requires torch and transformers; "
                "they are not product dependencies"
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
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                use_fast=False,
                **load_kwargs,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        except OSError as exc:
            mode = "download was allowed" if self.allow_download else "cache-only mode is active"
            raise RuntimeError(
                f"could not load pinned model {self.model_id}@{self.revision}; {mode}"
            ) from exc
        if int(getattr(model.config, "num_labels", 0) or 0) != 1:
            raise RuntimeError("cross-encoder benchmark requires one scalar relevance logit")
        if self.dtype == "float16":
            model = model.half()
        model = model.to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._load_duration_ms = (time.perf_counter() - started) * 1000.0

    def score_pairs(self, pairs: Sequence[Pair]) -> Sequence[float]:
        clean_pairs = [(str(query or ""), str(passage or "")) for query, passage in pairs]
        if not clean_pairs:
            return []
        self._load()
        torch = self._torch
        scores: list[float] = []
        started = time.perf_counter()
        truncated = 0
        max_observed = 0
        for offset in range(0, len(clean_pairs), self.batch_size):
            batch = clean_pairs[offset : offset + self.batch_size]
            queries = [query for query, _passage in batch]
            passages = [passage for _query, passage in batch]
            raw_tokens = self._tokenizer(
                queries,
                passages,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                verbose=False,
            )["input_ids"]
            raw_lengths = [len(row) for row in raw_tokens]
            max_observed = max(max_observed, max(raw_lengths, default=0))
            truncated += sum(1 for length in raw_lengths if length > self.max_length)
            encoded = self._tokenizer(
                queries,
                passages,
                max_length=self.max_length,
                padding=True,
                truncation="only_second",
                return_tensors="pt",
                verbose=False,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits.float().reshape(-1)
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        self._pair_count += len(clean_pairs)
        self._truncated_pair_count += truncated
        self._max_observed_tokens = max(self._max_observed_tokens, max_observed)
        self._score_duration_ms += (time.perf_counter() - started) * 1000.0
        return scores

    def runtime_info(self) -> dict[str, Any]:
        torch_version = str(getattr(self._torch, "__version__", "")) if self._torch else ""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "contract_identity": self.contract_identity,
            "architecture": "transformers_sequence_classification",
            "tokenizer": "sentencepiece_slow",
            "truncation": "only_second",
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "device": self._device or self.requested_device,
            "dtype": self.dtype,
            "allow_download": self.allow_download,
            "trust_remote_code": False,
            "load_duration_ms": round(self._load_duration_ms, 3),
            "pair_count": self._pair_count,
            "truncated_pair_count": self._truncated_pair_count,
            "max_observed_tokens": self._max_observed_tokens,
            "score_duration_ms": round(self._score_duration_ms, 3),
            "torch_version": torch_version,
        }
