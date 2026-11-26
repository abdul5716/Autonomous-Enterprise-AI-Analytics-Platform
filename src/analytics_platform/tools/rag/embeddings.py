"""Local CPU embeddings (no API cost) and a deterministic stand-in for tests."""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from analytics_platform.config import Settings

TOKEN = re.compile(r"[a-z0-9$%.,]+")


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """bge-small-en-v1.5 through ONNX Runtime: 384 dimensions, ~130 MB, a few ms per chunk."""

    def __init__(self, settings: Settings) -> None:
        from fastembed import TextEmbedding

        self.dim = settings.embedding_dim
        self._model = TextEmbedding(settings.embedding_model, cache_dir=settings.model_cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed([text]))).tolist()  # type: ignore[no-any-return]


class HashingEmbedder:
    """Bag-of-words hashed into a fixed vector. Only lexical overlap counts; for tests."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.strip(".,").encode(), digest_size=4).digest()
            vec[int.from_bytes(digest) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
