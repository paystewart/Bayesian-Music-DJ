from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None


@dataclass
class LabelIndex:
    name: str
    labels: list[str]
    embeddings: np.ndarray | None = None
    vectorizer: Any | None = None
    matrix: Any | None = None


class SemanticEmbedder:
    """
    Embedding wrapper with two backends:
    1) sentence-transformers/all-MiniLM-L6-v2 (preferred)
    2) TF-IDF fallback when sentence-transformers is unavailable
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | Path = ".cache/music_query_parser",
        allow_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.models_dir = self.cache_dir / "models"
        self.indices_dir = self.cache_dir / "indices"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        self.backend = "sentence-transformers"
        self.model = None
        if SentenceTransformer is None:
            if not allow_fallback:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
            if TfidfVectorizer is None:
                raise ImportError(
                    "Missing both sentence-transformers and scikit-learn. "
                    "Install sentence-transformers to enable semantic matching."
                )
            self.backend = "tfidf-fallback"
        else:
            self.model = SentenceTransformer(model_name, cache_folder=str(self.models_dir))

    def build_index(self, name: str, labels: list[str]) -> LabelIndex:
        cleaned = [label.strip().lower() for label in labels if label.strip()]
        digest = hashlib.sha256(
            f"{self.backend}|{self.model_name}|{name}|{'|'.join(cleaned)}".encode("utf-8")
        ).hexdigest()[:16]

        if self.backend == "sentence-transformers":
            cache_path = self.indices_dir / f"{name}_{digest}.npy"
            if cache_path.exists():
                embeddings = np.load(cache_path)
            else:
                embeddings = self._encode(cleaned)
                np.save(cache_path, embeddings)
            return LabelIndex(name=name, labels=cleaned, embeddings=embeddings)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        matrix = vectorizer.fit_transform(cleaned)
        return LabelIndex(name=name, labels=cleaned, vectorizer=vectorizer, matrix=matrix)

    def similarity_search(
        self,
        text: str,
        index: LabelIndex,
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> list[tuple[str, float]]:
        if not text.strip():
            return []

        if self.backend == "sentence-transformers":
            if index.embeddings is None:
                return []
            query = self._encode([text])[0]
            scores = index.embeddings @ query
        else:
            if index.vectorizer is None or index.matrix is None:
                return []
            query = index.vectorizer.transform([text])
            scores = (index.matrix @ query.T).toarray().ravel()

        ranked = np.argsort(scores)[::-1]
        out: list[tuple[str, float]] = []
        for idx in ranked:
            score = float(scores[idx])
            if score < min_score:
                continue
            out.append((index.labels[idx], score))
            if len(out) >= top_k:
                break
        return out

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Transformer backend unavailable.")
        arr = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
