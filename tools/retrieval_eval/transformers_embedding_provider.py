from __future__ import annotations

# Compatibility import for frozen evaluation tools. The reusable cache-only
# implementation lives in kb so product runtime code never depends on tools.
from kb.transformers_embedding_provider import (
    TransformersMeanPoolEmbeddingProvider,
    TransformersPoolingEmbeddingProvider,
    pool_hidden_states,
)


__all__ = [
    "TransformersMeanPoolEmbeddingProvider",
    "TransformersPoolingEmbeddingProvider",
    "pool_hidden_states",
]
