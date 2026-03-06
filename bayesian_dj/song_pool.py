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
        return 0.75 * genre_scores + 0.25 * global_scores

    def mark_used(self, pool_idx: int) -> None:
        self._available[pool_idx] = False

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
