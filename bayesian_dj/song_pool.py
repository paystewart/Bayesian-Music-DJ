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
EXCLUDED_FAMILY_PATTERNS = (
    r"\bcocomelon\b",
    r"\bbaby shark\b",
    r"\bwheels on the bus\b",
    r"\bnursery rhyme",
    r"\bnursery rhymes\b",
    r"\bkids\b",
    r"\bchildren(?:'s)?\b",
    r"\btoddler\b",
    r"\blullaby\b",
    r"\bsing along\b",
    r"\bsing-along\b",
    r"\bsuper simple songs\b",
    r"\blittle baby bum\b",
    r"\bmother goose\b",
    r"\bpeppa pig\b",
    r"\bdisney junior\b",
)


def filter_non_adult_catalog_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    text = (
        df["track_name"].fillna("").astype(str)
        + " "
        + df["artists"].fillna("").astype(str)
        + " "
        + df["album_name"].fillna("").astype(str)
        + " "
        + df["track_genre"].fillna("").astype(str)
    ).str.lower()
    pattern = "|".join(EXCLUDED_FAMILY_PATTERNS)
    mask = ~text.str.contains(pattern, regex=True, na=False)
    return df.loc[mask].reset_index(drop=True)


@dataclass
class SongInfo:
    pool_idx: int
    track_id: str
    track_name: str
    artists: str
    artist_ids: str
    album_name: str
    genre: str
    popularity: int
    artist_popularity: float
    release_year: int
    source_type: str
    prompt_score: float
    novelty_score: float
    is_saved: bool
    is_recent: bool
    is_top_track: bool
    features: dict[str, float]
    raw_tempo: float
    raw_loudness: float


class SongPool:
    """Manages the candidate pool for a single recommendation session.

    Can be initialised from a CSV file (legacy CLI path) or from a
    pre-built DataFrame supplied by the Spotify-backed UI path via
    ``SongPool.from_songs(df)``.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._df = filter_non_adult_catalog_df(pd.read_csv(csv_path))
        self._normalize()
        self._available: np.ndarray = np.ones(len(self._df), dtype=bool)
        self._external_bias = np.zeros(len(self._df), dtype=np.float64)

    @classmethod
    def from_songs(cls, df: pd.DataFrame) -> "SongPool":
        """Build a SongPool from a pre-built DataFrame (no CSV needed).

        The DataFrame must contain AUDIO_FEATURES columns plus:
        track_id, track_name, artists, album_name, track_genre, popularity.
        Missing columns are filled with sensible defaults.
        """
        pool = cls.__new__(cls)
        pool._df = filter_non_adult_catalog_df(df.reset_index(drop=True).copy())

        for col in ("track_id", "track_name", "artists", "artist_ids", "album_name", "track_genre", "source_type"):
            if col not in pool._df.columns:
                pool._df[col] = ""
        if "popularity" not in pool._df.columns:
            pool._df["popularity"] = 50
        for col, default in (
            ("artist_popularity", 0.0),
            ("release_year", 0),
            ("prompt_score", 0.0),
            ("novelty_score", 0.5),
            ("is_saved", False),
            ("is_recent", False),
            ("is_top_track", False),
        ):
            if col not in pool._df.columns:
                pool._df[col] = default
        for feat in AUDIO_FEATURES:
            if feat not in pool._df.columns:
                pool._df[feat] = 0.0

        if pool._df.empty:
            pool._raw_loudness = np.array([], dtype=np.float64)
            pool._raw_tempo = np.array([], dtype=np.float64)
            pool._genre_popularity_score = pd.Series(dtype=np.float64)
            pool._global_popularity_score = pd.Series(dtype=np.float64)
            pool._track_signature = pd.Series(dtype=str)
        else:
            pool._normalize()

        pool._available = np.ones(len(pool._df), dtype=bool)
        pool._external_bias = np.zeros(len(pool._df), dtype=np.float64)
        return pool

    def _normalize(self) -> None:
        self._raw_loudness = self._df["loudness"].values.copy()
        self._raw_tempo = self._df["tempo"].values.copy()
        self._genre_popularity_score = (
            self._df.groupby("track_genre")["popularity"].rank(method="average", pct=True)
        ).astype(np.float64)
        self._global_popularity_score = (
            self._df["popularity"].rank(method="average", pct=True)
        ).astype(np.float64)
        self._track_signature = (
            self._df["track_name"].fillna("").astype(str).str.lower().str.strip()
            + "__"
            + self._df["artists"].fillna("").astype(str).str.lower().str.strip()
        )

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
        matched = genre_col.isin(normalised)
        if matched.sum() == 0:
            return
        unknown = genre_col.eq("")
        # Spotify history/library rows often arrive without a resolved genre.
        # Keep those as fallback candidates instead of collapsing the pool.
        mask = matched | unknown
        self._available &= mask.to_numpy()

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
        if pool_idx < 0 or pool_idx >= len(self._available):
            raise IndexError(f"pool_idx {pool_idx} is out of bounds for pool of size {len(self._available)}")
        self._available[pool_idx] = False

    def resolve_pool_index(
        self,
        *,
        pool_idx: int | None = None,
        track_id: str = "",
        track_name: str = "",
        artists: str = "",
    ) -> int | None:
        if pool_idx is not None and 0 <= int(pool_idx) < len(self._df):
            row = self._df.iloc[int(pool_idx)]
            if track_id and str(row.get("track_id", "")) == str(track_id):
                return int(pool_idx)
            if not track_id:
                signature = f"{str(row.get('track_name', '')).strip().lower()}__{str(row.get('artists', '')).strip().lower()}"
                target = f"{str(track_name).strip().lower()}__{str(artists).strip().lower()}"
                if signature == target:
                    return int(pool_idx)

        if track_id:
            matches = self._df["track_id"].fillna("").astype(str) == str(track_id)
            if matches.any():
                return int(matches.idxmax())

        target_signature = f"{str(track_name).strip().lower()}__{str(artists).strip().lower()}"
        if target_signature.strip("_"):
            matches = self._track_signature == target_signature
            if matches.any():
                return int(matches.idxmax())
        return None

    def mark_song_used(
        self,
        *,
        pool_idx: int | None = None,
        track_id: str = "",
        track_name: str = "",
        artists: str = "",
    ) -> bool:
        resolved_idx = self.resolve_pool_index(
            pool_idx=pool_idx,
            track_id=track_id,
            track_name=track_name,
            artists=artists,
        )
        if resolved_idx is None:
            return False
        self._available[resolved_idx] = False
        return True

    def mark_used_track_ids(self, track_ids: set[str]) -> None:
        if not track_ids:
            return
        normalized = {str(track_id) for track_id in track_ids if str(track_id)}
        if not normalized:
            return
        mask = self._df["track_id"].fillna("").astype(str).isin(normalized)
        if mask.any():
            self._available &= ~mask.to_numpy()

    def mark_used_track_signatures(self, signatures: set[str]) -> None:
        if not signatures:
            return
        normalized = {
            str(signature).strip().lower()
            for signature in signatures
            if str(signature).strip()
        }
        if not normalized:
            return
        mask = self._track_signature.isin(normalized)
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
            artist_ids=str(row.get("artist_ids", "")),
            album_name=str(row["album_name"]),
            genre=str(row["track_genre"]),
            popularity=int(row["popularity"]),
            artist_popularity=float(row.get("artist_popularity", 0.0)),
            release_year=int(row.get("release_year", 0) or 0),
            source_type=str(row.get("source_type", "")),
            prompt_score=float(row.get("prompt_score", 0.0)),
            novelty_score=float(row.get("novelty_score", 0.5)),
            is_saved=bool(row.get("is_saved", False)),
            is_recent=bool(row.get("is_recent", False)),
            is_top_track=bool(row.get("is_top_track", False)),
            features=features,
            raw_tempo=float(self._raw_tempo[pool_idx]),
            raw_loudness=float(self._raw_loudness[pool_idx]),
        )
