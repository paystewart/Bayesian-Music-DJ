from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscoveryWeights:
    posterior: float = 0.18
    prompt: float = 0.24
    audio: float = 0.10
    artist: float = 0.06
    genre: float = 0.06
    popularity: float = 0.08
    novelty: float = 0.28
    related: float = 0.10
    saved_penalty: float = 0.20
    recent_penalty: float = 0.16
    top_penalty: float = 0.14
    shown_penalty: float = 0.50
    repeat_artist_penalty: float = 0.10


DEFAULT_DISCOVERY_WEIGHTS = DiscoveryWeights()


def normalize_label(value: str) -> str:
    lowered = (value or "").strip().lower().replace("-", " ").replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _coerce_beta_entry(entry: Any) -> tuple[float, float]:
    if isinstance(entry, dict):
        alpha = float(entry.get("alpha", 1.0) or 1.0)
        beta = float(entry.get("beta", 1.0) or 1.0)
        return max(alpha, 0.05), max(beta, 0.05)
    return 1.0, 1.0


def beta_mean(bucket: dict[str, Any], key: str, default_alpha: float = 1.0, default_beta: float = 1.0) -> float:
    entry = bucket.get(normalize_label(key))
    if not isinstance(entry, dict):
        return default_alpha / (default_alpha + default_beta)
    alpha, beta = _coerce_beta_entry(entry)
    return alpha / (alpha + beta)


def update_beta_bucket(
    bucket: dict[str, Any],
    values: list[str],
    *,
    liked: bool,
    amount: float,
    decay: float = 0.995,
) -> None:
    for raw_value in values:
        key = normalize_label(raw_value)
        if not key:
            continue
        alpha, beta = _coerce_beta_entry(bucket.get(key, {}))
        alpha *= decay
        beta *= decay
        if liked:
            alpha += amount
        else:
            beta += amount
        bucket[key] = {"alpha": round(alpha, 4), "beta": round(beta, 4)}


def mean_audio_vector(catalog: pd.DataFrame, weighted_tracks: list[dict[str, Any]], audio_features: list[str]) -> np.ndarray | None:
    if catalog.empty or not weighted_tracks:
        return None

    frame = catalog.copy()
    frame["_sig"] = (
        frame["track_name"].fillna("").astype(str).str.lower().str.strip()
        + "__"
        + frame["artists"].fillna("").astype(str).str.lower().str.strip()
    )
    lookup = frame.set_index("_sig")

    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for item in weighted_tracks:
        track_name = str(item.get("track_name", "") or "").strip().lower()
        artists = str(item.get("artists", "") or "").strip().lower()
        signature = f"{track_name}__{artists}"
        if signature not in lookup.index:
            continue
        row = lookup.loc[signature]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        vectors.append(row[audio_features].to_numpy(dtype=float))
        weights.append(float(item.get("weight", 1.0) or 1.0))
    if not vectors:
        return None
    return np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights, dtype=float))


def audio_similarity_scores(catalog: pd.DataFrame, center: np.ndarray | None, audio_features: list[str]) -> pd.Series:
    if catalog.empty or center is None:
        return pd.Series(0.0, index=catalog.index, dtype=float)
    matrix = catalog[audio_features].to_numpy(dtype=float)
    distances = np.abs(matrix - center).mean(axis=1)
    return pd.Series(np.clip(1.0 - distances, 0.0, 1.0), index=catalog.index, dtype=float)


def discovery_score_frame(
    catalog: pd.DataFrame,
    profile: dict[str, Any],
    prompt_terms: list[str],
    audio_features: list[str],
    *,
    weights: DiscoveryWeights = DEFAULT_DISCOVERY_WEIGHTS,
) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame(index=catalog.index)

    frame = pd.DataFrame(index=catalog.index)
    artist_bucket = profile.get("artist_posterior", {})
    genre_bucket = profile.get("genre_posterior", {})
    track_bucket = profile.get("track_posterior", {})
    novelty_state = profile.get("novelty_posterior", {"alpha": 6.0, "beta": 4.0})
    popularity_state = profile.get("popularity_posterior", {"alpha": 5.5, "beta": 4.5})
    novelty_pref = beta_mean({"novelty": novelty_state}, "novelty", 6.0, 4.0)
    popularity_pref = beta_mean({"mainstream": popularity_state}, "mainstream", 5.5, 4.5)

    artist_col = catalog["artists"].fillna("").astype(str).str.lower()
    genre_col = catalog["track_genre"].fillna("").astype(str).str.lower()
    track_col = catalog["track_name"].fillna("").astype(str).str.lower()
    source_col = catalog.get("source_type", pd.Series("", index=catalog.index)).fillna("").astype(str).str.lower()

    frame["artist_affinity"] = 0.0
    for artist, stats in artist_bucket.items():
        mean = beta_mean({artist: stats}, artist)
        if mean <= 0.5:
            continue
        frame["artist_affinity"] += artist_col.str.contains(re.escape(artist), case=False, na=False).astype(float) * (mean - 0.5) * 2.0

    frame["genre_affinity"] = 0.0
    for genre, stats in genre_bucket.items():
        mean = beta_mean({genre: stats}, genre)
        if mean <= 0.5:
            continue
        frame["genre_affinity"] += genre_col.str.contains(re.escape(genre), case=False, na=False).astype(float) * (mean - 0.5) * 2.0

    frame["track_affinity"] = 0.0
    for track_name, stats in list(track_bucket.items())[:24]:
        mean = beta_mean({track_name: stats}, track_name)
        if mean <= 0.5:
            continue
        frame["track_affinity"] += track_col.str.contains(re.escape(track_name), case=False, na=False).astype(float) * (mean - 0.5) * 1.3

    positive_examples = profile.get("recent_positive_examples", [])[:10]
    negative_examples = profile.get("recent_negative_examples", [])[:10]
    positive_center = mean_audio_vector(catalog, positive_examples, audio_features)
    negative_center = mean_audio_vector(catalog, negative_examples, audio_features)
    frame["audio_match"] = audio_similarity_scores(catalog, positive_center, audio_features)
    if negative_center is not None:
        frame["audio_match"] -= audio_similarity_scores(catalog, negative_center, audio_features) * 0.45
        frame["audio_match"] = frame["audio_match"].clip(lower=0.0)

    frame["prompt_match"] = catalog.get("prompt_score", pd.Series(0.0, index=catalog.index)).fillna(0.0).astype(float)
    prompt_text = " ".join(prompt_terms).strip().lower()
    if prompt_text:
        prompt_hits = (
            track_col.str.contains(re.escape(prompt_text), case=False, na=False).astype(float)
            + artist_col.str.contains(re.escape(prompt_text), case=False, na=False).astype(float)
            + genre_col.str.contains(re.escape(prompt_text), case=False, na=False).astype(float)
        ).clip(upper=1.0)
        frame["prompt_match"] = np.maximum(frame["prompt_match"], prompt_hits)

    popularity_pct = catalog["popularity"].rank(method="average", pct=True).astype(float)
    artist_popularity = catalog.get("artist_popularity", pd.Series(0.0, index=catalog.index)).fillna(0.0).astype(float) / 100.0
    frame["popularity_quality"] = (0.68 * popularity_pct + 0.32 * artist_popularity) * (0.75 + 0.25 * popularity_pref)

    source_bonus = pd.Series(0.0, index=catalog.index, dtype=float)
    source_bonus += source_col.str.contains("related_artist", case=False, na=False).astype(float) * 0.95
    source_bonus += source_col.str.contains("seed_artist_top", case=False, na=False).astype(float) * 0.82
    source_bonus += source_col.str.contains("recommendation", case=False, na=False).astype(float) * 0.72
    source_bonus += source_col.str.contains("search", case=False, na=False).astype(float) * 0.52
    frame["related_bonus"] = source_bonus.clip(upper=1.0)

    novelty_raw = catalog.get("novelty_score", pd.Series(0.5, index=catalog.index)).fillna(0.5).astype(float)
    frame["novelty_bonus"] = (0.45 + novelty_pref) * novelty_raw

    frame["saved_penalty"] = catalog.get("is_saved", pd.Series(False, index=catalog.index)).astype(float)
    frame["recent_penalty"] = catalog.get("is_recent", pd.Series(False, index=catalog.index)).astype(float)
    frame["top_penalty"] = catalog.get("is_top_track", pd.Series(False, index=catalog.index)).astype(float)

    frame["discovery_score"] = (
        weights.artist * frame["artist_affinity"]
        + weights.genre * frame["genre_affinity"]
        + 0.04 * frame["track_affinity"]
        + weights.audio * frame["audio_match"]
        + weights.prompt * frame["prompt_match"]
        + weights.popularity * frame["popularity_quality"]
        + weights.related * frame["related_bonus"]
        + weights.novelty * frame["novelty_bonus"]
        - weights.saved_penalty * frame["saved_penalty"]
        - weights.recent_penalty * frame["recent_penalty"]
        - weights.top_penalty * frame["top_penalty"]
    )
    return frame
