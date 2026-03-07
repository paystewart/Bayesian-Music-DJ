from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

LOUDNESS_MIN, LOUDNESS_MAX = -50.0, 5.0
TEMPO_MIN, TEMPO_MAX = 0.0, 250.0


@dataclass
class SongInfo:
    pool_idx: int
    track_id: str
    track_name: str
    artists: str
    album_name: str
    genre: str
    popularity: int
    features: dict[str, float]
    raw_tempo: float
    raw_loudness: float


class SongPool:
    """Loads the kaggle dataset, normalises audio features, and manages the
    candidate pool for a single recommendation session."""

    def __init__(self, csv_path: str | Path) -> None:
        self._df = pd.read_csv(csv_path)
        self._normalize()
        self._available: np.ndarray = np.ones(len(self._df), dtype=bool)
        self._external_bias = np.zeros(len(self._df), dtype=np.float64)

    def _normalize(self) -> None:
        self._raw_loudness = self._df["loudness"].values.copy()
        self._raw_tempo = self._df["tempo"].values.copy()
        self._genre_popularity_score = (
            self._df.groupby("track_genre")["popularity"].rank(method="average", pct=True)
        ).astype(np.float64)
        self._global_popularity_score = (
            self._df["popularity"].rank(method="average", pct=True)
        ).astype(np.float64)

        self._df["loudness"] = (
            (self._df["loudness"] - LOUDNESS_MIN) / (LOUDNESS_MAX - LOUDNESS_MIN)
        ).clip(0.0, 1.0)
        self._df["tempo"] = (
            (self._df["tempo"] - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
        ).clip(0.0, 1.0)

    def filter_by_genres(self, genres: list[str]) -> None:
        """Keep only rows whose track_genre matches at least one of *genres*.
        Matching is case-insensitive and tolerates hyphens vs spaces."""
        if not genres:
            return
        genre_col = self._df["track_genre"].str.lower().str.replace("-", " ")
        normalised = {g.lower().replace("-", " ") for g in genres}
        mask = genre_col.isin(normalised)
        if mask.sum() == 0:
            return
        self._available &= mask.values

    @property
    def n_available(self) -> int:
        return int(self._available.sum())

    def get_feature_matrix(self) -> np.ndarray:
        """Return (n_available, 10) matrix: bias column + 9 audio features."""
        sub = self._df.loc[self._available, AUDIO_FEATURES].values.astype(np.float64)
        bias = np.ones((sub.shape[0], 1), dtype=np.float64)
        return np.hstack([bias, sub])

    def available_indices(self) -> np.ndarray:
        return np.where(self._available)[0]

    def get_popularity_scores(self) -> np.ndarray:
        """Return popularity prior for available songs, favoring well-known tracks
        within the current genre while still considering global popularity."""
        genre_scores = self._genre_popularity_score.loc[self._available].to_numpy()
        global_scores = self._global_popularity_score.loc[self._available].to_numpy()
        return 0.55 * genre_scores + 0.45 * global_scores

    def set_external_bias(self, scores: np.ndarray) -> None:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.shape[0] != len(self._df):
            raise ValueError("External bias length must match catalog length.")
        finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(finite.min())
        hi = float(finite.max())
        if hi - lo < 1e-9:
            self._external_bias = np.zeros(len(self._df), dtype=np.float64)
            return
        self._external_bias = (finite - lo) / (hi - lo)

    def get_external_bias_scores(self) -> np.ndarray:
        return self._external_bias[self._available]

    def mark_used(self, pool_idx: int) -> None:
        self._available[pool_idx] = False

    def mark_used_track_ids(self, track_ids: set[str]) -> None:
        if not track_ids:
            return
        normalized = {str(track_id) for track_id in track_ids if str(track_id)}
        if not normalized:
            return
        mask = self._df["track_id"].fillna("").astype(str).isin(normalized)
        if mask.any():
            self._available &= ~mask.to_numpy()

    def get_song_info(self, pool_idx: int) -> SongInfo:
        row = self._df.iloc[pool_idx]
        features = {f: float(row[f]) for f in AUDIO_FEATURES}
        return SongInfo(
            pool_idx=pool_idx,
            track_id=str(row["track_id"]),
            track_name=str(row["track_name"]),
            artists=str(row["artists"]),
            album_name=str(row["album_name"]),
            genre=str(row["track_genre"]),
            popularity=int(row["popularity"]),
            features=features,
            raw_tempo=float(self._raw_tempo[pool_idx]),
            raw_loudness=float(self._raw_loudness[pool_idx]),
        )
