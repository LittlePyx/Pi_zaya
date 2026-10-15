from __future__ import annotations

import pytest

from tools.retrieval_eval.run_dense_research_qa_benchmark import _parse_source_safe_config
from tools.retrieval_eval.transformers_embedding_provider import (
    TransformersMeanPoolEmbeddingProvider,
    TransformersPoolingEmbeddingProvider,
)
from tools.retrieval_eval.transformers_cross_encoder_provider import (
    TransformersSequenceClassificationReranker,
)


def test_transformers_benchmark_provider_fingerprint_covers_encoding_contract() -> None:
    common = {
        "model_id": "intfloat/multilingual-e5-base",
        "revision": "a" * 40,
        "dimension": 768,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 8,
        "device": "cpu",
        "dtype": "float32",
    }
    base = TransformersMeanPoolEmbeddingProvider(max_length=512, **common)
    changed = TransformersMeanPoolEmbeddingProvider(max_length=256, **common)

    assert base.descriptor.kind == "local"
    assert base.descriptor.revision == "a" * 40
    assert base.descriptor.dimension == 768
    assert base.descriptor.identity != changed.descriptor.identity
    assert base.runtime_info()["allow_download"] is False


def test_transformers_benchmark_provider_rejects_ambiguous_runtime_contract() -> None:
    with pytest.raises(ValueError, match="dtype"):
        TransformersMeanPoolEmbeddingProvider(
            model_id="model",
            revision="revision",
            dimension=2,
            query_prefix="query: ",
            passage_prefix="passage: ",
            max_length=32,
            batch_size=1,
            dtype="auto",
        )


def test_transformers_benchmark_provider_freezes_pooling_and_padding() -> None:
    common = {
        "model_id": "model",
        "revision": "a" * 40,
        "dimension": 4,
        "query_prefix": "",
        "passage_prefix": "",
        "max_length": 32,
        "batch_size": 1,
        "device": "cpu",
        "dtype": "float32",
    }
    cls_provider = TransformersPoolingEmbeddingProvider(
        pooling="cls", padding_side="right", **common
    )
    last_provider = TransformersPoolingEmbeddingProvider(
        pooling="last_token", padding_side="left", **common
    )
    assert cls_provider.descriptor.identity != last_provider.descriptor.identity
    assert cls_provider.runtime_info()["pooling"] == "cls"
    assert last_provider.runtime_info()["padding_side"] == "left"

    with pytest.raises(ValueError, match="pooling"):
        TransformersPoolingEmbeddingProvider(pooling="max", **common)


def test_source_safe_config_parser_rejects_ambiguous_values() -> None:
    assert _parse_source_safe_config("2:5") == (2.0, 5)
    with pytest.raises(Exception, match="SPARSE_WEIGHT:GATE_DEPTH"):
        _parse_source_safe_config("2")




def test_cross_encoder_benchmark_contract_is_pinned_and_cache_first() -> None:
    common = {
        "model_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "revision": "b" * 40,
        "batch_size": 8,
        "device": "cpu",
        "dtype": "float32",
    }
    base = TransformersSequenceClassificationReranker(max_length=512, **common)
    changed = TransformersSequenceClassificationReranker(max_length=256, **common)

    assert base.contract_identity != changed.contract_identity
    assert base.runtime_info()["allow_download"] is False
    assert base.runtime_info()["trust_remote_code"] is False
