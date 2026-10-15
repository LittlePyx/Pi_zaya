from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class TransformersNliProvider:
    """Benchmark-only, pinned local NLI sequence classifier."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        max_length: int,
        batch_size: int,
        device: str,
        allow_download: bool,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.requested_device = str(device)
        self.allow_download = bool(allow_download)
        self.cache_dir = cache_dir.resolve() if cache_dir else None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = ""
        self._load_duration_ms = 0.0
        self._score_duration_ms = 0.0
        self._pair_count = 0
        self._truncated_pair_count = 0
        self._max_observed_tokens = 0
        self._label_order: list[str] = []

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
            raise RuntimeError("local NLI benchmark requires torch and transformers") from exc
        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        load_kwargs = {
            "revision": self.revision,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "local_files_only": not self.allow_download,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            use_safetensors=True,
            **load_kwargs,
        )
        label_order = [
            str(model.config.id2label[index]).strip().lower()
            for index in range(int(model.config.num_labels))
        ]
        if label_order != ["entailment", "neutral", "contradiction"]:
            raise ValueError(f"unexpected NLI label order: {label_order}")
        model = model.float().to(device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._label_order = label_order
        self._load_duration_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        self._load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        started = time.perf_counter()
        output: list[dict[str, Any]] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset : offset + self.batch_size]
            premises = [premise for premise, _claim in batch]
            claims = [claim for _premise, claim in batch]
            raw = tokenizer(
                premises,
                claims,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
            raw_lengths = [len(ids) for ids in raw["input_ids"]]
            self._pair_count += len(batch)
            self._truncated_pair_count += sum(length > self.max_length for length in raw_lengths)
            self._max_observed_tokens = max([self._max_observed_tokens, *raw_lengths])
            encoded = tokenizer(
                premises,
                claims,
                add_special_tokens=True,
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
                verbose=False,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu()
            for row in probabilities.tolist():
                output.append(
                    {
                        "scores": {
                            label: round(float(score), 6)
                            for label, score in zip(self._label_order, row)
                        }
                    }
                )
        self._score_duration_ms += (time.perf_counter() - started) * 1000.0
        return output

    def runtime_info(self) -> dict[str, Any]:
        torch_version = str(getattr(self._torch, "__version__", "")) if self._torch else ""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "architecture": type(self._model).__name__ if self._model is not None else "",
            "tokenizer": type(self._tokenizer).__name__ if self._tokenizer is not None else "",
            "label_order": list(self._label_order),
            "max_length": self.max_length,
            "truncation": "only_first",
            "batch_size": self.batch_size,
            "device": self._device,
            "precision": "float32",
            "allow_download": self.allow_download,
            "trust_remote_code": False,
            "safe_weights_required": True,
            "load_duration_ms": round(self._load_duration_ms, 3),
            "pair_count": self._pair_count,
            "truncated_pair_count": self._truncated_pair_count,
            "max_observed_tokens": self._max_observed_tokens,
            "score_duration_ms": round(self._score_duration_ms, 3),
            "torch_version": torch_version,
            "product_enabled": False,
        }
