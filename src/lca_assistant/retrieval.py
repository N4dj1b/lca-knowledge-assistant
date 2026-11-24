"""TF-IDF retrieval baseline with optional SentenceTransformers dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import DocumentChunk


@dataclass
class SearchHit:
    chunk: DocumentChunk
    score: float


class Retriever:
    def __init__(self, backend: str = "tfidf", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.backend = backend.lower()
        self.model_name = model_name
        self.chunks: list[DocumentChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.encoder = None
        self.vectors: np.ndarray | None = None
        self.backend_used = "tfidf"

    def fit(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a retrieval index without document chunks.")
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        if self.backend in {"sentence-transformers", "embeddings", "dense"}:
            try:
                from sentence_transformers import SentenceTransformer

                self.encoder = SentenceTransformer(self.model_name)
                self.vectors = self.encoder.encode(texts, normalize_embeddings=True)
                self.backend_used = "sentence-transformers"
                return
            except ImportError:
                # Keep the core MVP usable without the optional, larger dependency.
                self.encoder = None
            except Exception as exc:  # model download/cache issues should not break the MVP
                self.encoder = None
                self.backend_used = f"tfidf (embedding fallback: {type(exc).__name__})"

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(texts)
        if self.backend_used != "tfidf":
            return
        self.backend_used = "tfidf"

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if not self.chunks or not query.strip():
            return []
        if self.encoder is not None and self.vectors is not None:
            query_vector = self.encoder.encode([query], normalize_embeddings=True)
            scores = np.asarray(self.vectors @ query_vector[0]).reshape(-1)
        else:
            if self.vectorizer is None or self.matrix is None:
                return []
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = np.argsort(scores)[::-1][: max(1, top_k)]
        return [SearchHit(self.chunks[int(index)], float(scores[int(index)])) for index in ranked if scores[int(index)] > 0]
