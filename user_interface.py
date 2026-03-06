from __future__ import annotations

import base64
import html
import json
import os
import re
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ModuleNotFoundError:
    px = None
    go = Any
    HAS_PLOTLY = False

from bayesian_dj.session import DJSession, DEFAULT_CSV
from bayesian_dj.model import BayesianLogisticRegression
from bayesian_dj.song_pool import AUDIO_FEATURES
from music_query_parser.parser import MusicQueryParser, QuerySpec

st.set_page_config(
    page_title="Bayesian DJ",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AUDIO_FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "loudness": "Loudness",
    "speechiness": "Speechiness",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "liveness": "Liveness",
    "valence": "Valence",
    "tempo": "Tempo",
}

PROMPT_PRESETS = [
    "Dreamy late-night indie with soft vocals and low energy",
    "High-energy house for a rooftop party with strong danceability",
    "Instrumental focus music that stays calm and not too upbeat",
    "Moody alt-pop similar to The 1975 but darker and more atmospheric",
]

REFINEMENT_PRESETS = [
    "Make it more soulful and less electronic",
    "Slow it down and keep it intimate",
    "Open up the genre filter but keep the mood warm",
    "Make it more acoustic and less dance-driven",
]

DATA_DIR = Path("data")
UI_STATE_PATH = DATA_DIR / "ui_user_state.json"

DATA_DIR.mkdir(exist_ok=True)


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_CSV)


@st.cache_resource
def get_parser() -> MusicQueryParser:
    return MusicQueryParser()


def default_ui_state() -> dict[str, Any]:
    return {
        "artist_affinity": {},
        "genre_affinity": {},
        "track_affinity": {},
        "liked_songs": [],
        "spotify_art_cache": {},
        "spotify_user_seeded": False,
        "spotify_user_summary": {},
    }


def load_ui_state() -> dict[str, Any]:
    if not UI_STATE_PATH.exists():
        return default_ui_state()

    with UI_STATE_PATH.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    defaults = default_ui_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def save_ui_state(state: dict[str, Any]) -> None:
    UI_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg: #0f6b47;
            --panel: rgba(255, 255, 255, 0.10);
            --panel-strong: rgba(255, 255, 255, 0.16);
            --ink: #f5f9ff;
            --muted: rgba(245, 249, 255, 0.72);
            --accent: #1ed760;
            --accent-soft: #88f0b2;
            --forest: #d9fbe5;
            --gold: #a7c3ff;
            --border: rgba(255, 255, 255, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(255,255,255,0.08), transparent 24%),
                linear-gradient(180deg, #178a57 0%, #116f48 42%, #0b5b3a 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        div[data-testid="collapsedControl"] {
            display: none;
        }

        [data-testid="stSidebar"] * {
            color: #f6efe4;
            font-family: "Space Grotesk", "Avenir Next", sans-serif;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #f3c6a6, #e39b6d);
            color: #1d2a26;
            border: 1px solid rgba(29, 42, 38, 0.18);
            box-shadow: 0 10px 24px rgba(15, 24, 22, 0.22);
        }

        [data-testid="stSidebar"] .stButton > button p,
        [data-testid="stSidebar"] .stButton > button span,
        [data-testid="stSidebar"] .stButton > button div {
            color: #1d2a26 !important;
        }

        [data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: rgba(255, 248, 239, 0.96);
            border: 1px solid rgba(29, 42, 38, 0.14);
            box-shadow: 0 12px 28px rgba(15, 24, 22, 0.18);
        }

        [data-testid="stSidebar"] div[data-testid="stMetric"] label,
        [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
        [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
            color: #1d2a26 !important;
        }

        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] .stCaptionContainer *,
        [data-testid="stSidebar"] ol,
        [data-testid="stSidebar"] li {
            color: #f4eee2 !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 920px;
        }

        h1, h2, h3 {
            font-family: "Fraunces", "Iowan Old Style", serif;
            letter-spacing: -0.02em;
            color: var(--ink);
        }

        p, li, div, label, span {
            font-family: "Space Grotesk", "Avenir Next", sans-serif;
        }

        .hero-card, .glass-card, .track-card, .metric-card {
            background: var(--panel);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            box-shadow: 0 18px 40px rgba(65, 49, 32, 0.10);
            border-radius: 24px;
        }

        .hero-card {
            padding: 1.8rem 1.8rem 1.4rem 1.8rem;
            background:
                linear-gradient(135deg, rgba(255, 248, 238, 0.94), rgba(244, 227, 204, 0.82)),
                radial-gradient(circle at top right, rgba(201, 91, 53, 0.18), transparent 22%);
        }

        .glass-card, .track-card {
            padding: 1.2rem 1.2rem 1rem 1.2rem;
        }

        .track-card {
            background:
                linear-gradient(180deg, rgba(255, 250, 244, 0.96), rgba(247, 240, 230, 0.90));
        }

        .metric-card {
            padding: 1rem 1.1rem;
            min-height: 118px;
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: var(--accent);
            font-weight: 700;
        }

        .hero-title {
            font-size: clamp(2.2rem, 4vw, 4.4rem);
            line-height: 0.94;
            margin: 0.45rem 0 0.8rem 0;
            max-width: 10ch;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1rem;
            max-width: 44rem;
            margin-bottom: 0;
        }

        .section-label {
            margin: 1rem 0 0.65rem 0;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            font-weight: 700;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.6rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(39, 76, 67, 0.10);
            color: var(--forest);
            font-size: 0.84rem;
            font-weight: 600;
        }

        .chip.warm {
            background: rgba(201, 91, 53, 0.12);
            color: var(--accent);
        }

        .chip.gold {
            background: rgba(184, 137, 39, 0.14);
            color: #7b5e1d;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            color: var(--ink);
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }

        .metric-subtext {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }

        .track-kicker {
            color: var(--accent);
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-size: 0.72rem;
        }

        .track-title {
            font-family: "Fraunces", "Iowan Old Style", serif;
            font-size: clamp(1.8rem, 2.8vw, 3rem);
            line-height: 1.02;
            margin: 0.35rem 0 0.45rem 0;
        }

        .track-meta {
            color: var(--muted);
            font-size: 1rem;
            margin-bottom: 0.8rem;
        }

        .assistant-note {
            background: rgba(39, 76, 67, 0.08);
            border-left: 4px solid var(--forest);
            color: var(--ink);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-top: 0.9rem;
        }

        .playlist-item {
            padding: 0.8rem 0;
            border-bottom: 1px solid rgba(34, 49, 43, 0.08);
        }

        .playlist-item:last-child {
            border-bottom: none;
        }

        .playlist-title {
            font-weight: 700;
            color: var(--ink);
        }

        .playlist-meta {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .empty-state {
            padding: 1.2rem 0.2rem 0.2rem 0.2rem;
            color: var(--muted);
        }

        .rail-shell {
            background: rgba(255, 249, 241, 0.84);
            border: 1px solid var(--border);
            box-shadow: 0 18px 40px rgba(65, 49, 32, 0.10);
            border-radius: 28px;
            padding: 1rem;
        }

        .rail-header {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }

        .rail-scroll {
            display: flex;
            gap: 1rem;
            overflow-x: auto;
            padding-bottom: 0.25rem;
            scroll-snap-type: x proximity;
        }

        .rail-card {
            min-width: 196px;
            max-width: 196px;
            scroll-snap-align: start;
        }

        .rail-cover {
            width: 196px;
            height: 196px;
            object-fit: cover;
            border-radius: 20px;
            display: block;
            box-shadow: 0 14px 34px rgba(23, 30, 28, 0.18);
            margin-bottom: 0.65rem;
        }

        .rail-title {
            color: var(--ink);
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.2rem;
        }

        .rail-subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.2;
            min-height: 2.2rem;
        }

        .rail-meta {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .deck-shell {
            background:
                linear-gradient(135deg, rgba(29, 42, 38, 0.96), rgba(42, 57, 52, 0.92));
            color: #f8f1e8;
            border-radius: 30px;
            padding: 1.2rem;
            box-shadow: 0 24px 48px rgba(23, 30, 28, 0.22);
        }

        .deck-shell .track-kicker,
        .deck-shell .track-meta,
        .deck-shell .track-title,
        .deck-shell .assistant-note {
            color: inherit;
        }

        .cover-caption {
            color: rgba(248, 241, 232, 0.78);
            font-size: 0.88rem;
            margin-top: 0.6rem;
        }

        .chat-shell {
            background: rgba(255, 249, 241, 0.84);
            border: 1px solid var(--border);
            border-radius: 28px;
            box-shadow: 0 18px 40px rgba(65, 49, 32, 0.10);
            padding: 1rem;
        }

        .status-banner {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            margin-top: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.16);
            font-weight: 600;
            line-height: 1.45;
        }

        .status-banner.success {
            background: rgba(245, 251, 247, 0.96);
            color: #103322;
        }

        .status-banner.info {
            background: rgba(241, 247, 255, 0.96);
            color: #17304f;
        }

        .status-banner.warning {
            background: rgba(255, 247, 232, 0.97);
            color: #5a3b11;
        }

        .playback-shell {
            background: rgba(14, 28, 22, 0.56);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 22px;
            padding: 0.9rem 1rem 1rem 1rem;
            margin: 0.85rem 0 1rem 0;
        }

        .playback-title {
            color: #f5f9ff;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }

        .playback-meta {
            color: rgba(245, 249, 255, 0.74);
            font-size: 0.92rem;
            margin-top: 0.55rem;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            border-radius: 20px;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            color: var(--ink);
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            padding: 0.65rem 1rem;
            font-weight: 700;
            font-family: "Space Grotesk", "Avenir Next", sans-serif;
            transition: transform 120ms ease, box-shadow 120ms ease;
            background: rgba(255, 255, 255, 0.10);
            color: #f5f9ff;
            box-shadow: 0 8px 24px rgba(7, 22, 63, 0.20);
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
        }

        div.stButton > button:focus,
        div.stButton > button:focus-visible {
            color: #f5f9ff;
            border-color: rgba(255, 255, 255, 0.36);
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }

        .stTextInput input, .stTextArea textarea {
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.10);
            color: #f5f9ff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("dj_session", None)
    st.session_state.setdefault("session_finished", False)
    st.session_state.setdefault("last_feedback", None)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("prompt_input", PROMPT_PRESETS[0])
    st.session_state.setdefault("ui_state", load_ui_state())
    st.session_state.setdefault("playlist_length", 12)
    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("speech_payload", None)
    st.session_state.setdefault("last_transition_round", 0)
    st.session_state.setdefault("latest_model_update", "")
    st.session_state.setdefault("playback_song", None)
    st.session_state.setdefault("playback_scored", False)


def reset_session() -> None:
    st.session_state["dj_session"] = None
    st.session_state["session_finished"] = False
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["is_playing"] = False
    st.session_state["speech_payload"] = None
    st.session_state["last_transition_round"] = 0
    st.session_state["latest_model_update"] = ""
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False


def session_complete(session: DJSession | None) -> bool:
    if session is None:
        return True
    if st.session_state.get("session_finished", False):
        return True
    if len(session.playlist) >= session.playlist_length:
        return True
    if session.pool.n_available == 0 and session._current_song is None:
        return True
    return False


def ensure_current_song(session: DJSession) -> None:
    if session is None or len(session.playlist) >= session.playlist_length:
        return
    if session._current_song is None:
        song = session.recommend_next()
        if song is None:
            st.session_state["session_finished"] = True


def clone_spec(spec: QuerySpec) -> QuerySpec:
    return QuerySpec(
        seed_track=spec.seed_track,
        seed_artists=list(spec.seed_artists),
        genres=list(spec.genres),
        moods=list(spec.moods),
        constraints=dict(spec.constraints),
        year_range=spec.year_range,
        playlist_length=spec.playlist_length,
        spotify_search_queries=list(spec.spotify_search_queries),
    )


def spec_feature_vector(song) -> np.ndarray:
    return np.array([1.0, *[song.features[name] for name in AUDIO_FEATURES]], dtype=np.float64)


def build_session_from_spec(
    spec: QuerySpec,
    playlist_length: int,
    prior_feedback: list[tuple[object, str]] | None = None,
) -> DJSession:
    session = DJSession(
        csv_path=DEFAULT_CSV,
        playlist_length=playlist_length,
        parser=get_parser(),
    )
    session.spec = clone_spec(spec)
    session.pool.filter_by_genres(session.spec.genres)
    session.model = BayesianLogisticRegression.from_constraints(dict(session.spec.constraints))
    session.model.snapshot()
    session.initial_candidate_count = session.pool.n_available

    for song, action in prior_feedback or []:
        x = spec_feature_vector(song)
        y = 1 if action == "play" else 0
        session.model.update(x, y)
        session.model.snapshot(x=x, y=y)
        session.playlist.append(song)
        session.actions.append(action)
        session.pool.mark_used(song.pool_idx)

    if len(session.playlist) < session.playlist_length and session.pool.n_available > 0:
        session.recommend_next()
    return session


def add_chat_message(role: str, content: str) -> None:
    st.session_state["chat_messages"].append({"role": role, "content": content})


def summarize_spec(spec: QuerySpec, session: DJSession) -> str:
    fragments = [f"I filtered toward **{getattr(session, 'initial_candidate_count', session.pool.n_available):,} candidate songs**."]
    fragments.append(f"Detected genres: {', '.join(spec.genres) if spec.genres else 'open search'}.")
    fragments.append(f"Moods: {', '.join(spec.moods) if spec.moods else 'no explicit mood locks'}.")
    if spec.constraints:
        fragments.append("Feature priors: " + ", ".join(format_constraints(spec)) + ".")
    else:
        fragments.append("Feature priors are currently broad, so the posterior has more room to learn from feedback.")
    if session._current_song is not None:
        fragments.append(
            f'Next recommendation: **{session._current_song.track_name}** by **{session._current_song.artists}**.'
        )
    return " ".join(fragments)


def parse_preference_text(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r",|\n", raw) if item.strip()]


def current_taste_profile() -> dict[str, Any]:
    return st.session_state["ui_state"]


def bump_affinity(bucket: dict[str, float], values: list[str], amount: float) -> None:
    for value in values:
        key = value.strip().lower()
        if not key:
            continue
        bucket[key] = bucket.get(key, 0.0) + amount


def infer_preferences_from_message(message: str, spec: QuerySpec) -> None:
    state = current_taste_profile()
    boost = 2.2 if any(
        phrase in message.lower()
        for phrase in ("usually listen to", "i listen to", "favorite", "love", "mostly listen to")
    ) else 1.0
    bump_affinity(state.setdefault("genre_affinity", {}), spec.genres, 1.4 * boost)
    bump_affinity(state.setdefault("artist_affinity", {}), spec.seed_artists, 1.8 * boost)
    if spec.seed_track:
        bump_affinity(state.setdefault("track_affinity", {}), [spec.seed_track], 1.6 * boost)
    save_ui_state(state)


def infer_preferences_from_song(song, played: bool) -> None:
    state = current_taste_profile()
    amount = 1.4 if played else -0.45
    artists = [artist.strip() for artist in song.artists.split(";") if artist.strip()]
    bump_affinity(state.setdefault("artist_affinity", {}), artists, amount)
    bump_affinity(state.setdefault("genre_affinity", {}), [song.genre], amount)
    bump_affinity(state.setdefault("track_affinity", {}), [song.track_name], amount * 0.8)
    save_ui_state(state)


def add_song_to_liked(song) -> None:
    state = current_taste_profile()
    liked = state.setdefault("liked_songs", [])
    payload = {
        "track_id": song.track_id,
        "track_name": song.track_name,
        "artists": song.artists,
        "album_name": song.album_name,
        "genre": song.genre,
    }
    if payload not in liked:
        liked.insert(0, payload)
        state["liked_songs"] = liked[:24]
        save_ui_state(state)


def serialize_song(song) -> dict[str, Any]:
    return {
        "pool_idx": song.pool_idx,
        "track_id": song.track_id,
        "track_name": song.track_name,
        "artists": song.artists,
        "album_name": song.album_name,
        "genre": song.genre,
        "popularity": song.popularity,
        "features": dict(song.features),
        "raw_tempo": song.raw_tempo,
        "raw_loudness": song.raw_loudness,
    }


def deserialize_song(song_data: dict[str, Any] | None):
    if not song_data:
        return None
    return SimpleNamespace(**song_data)


def top_affinity_items(bucket: dict[str, float], limit: int = 3) -> list[str]:
    ranked = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
    return [key for key, score in ranked if score > 0][:limit]


def preference_matches(catalog: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    if catalog.empty:
        return catalog.iloc[0:0]

    mask = pd.Series(False, index=catalog.index)

    for artist in top_affinity_items(profile.get("artist_affinity", {}), limit=4):
        artist_mask = catalog["artists"].str.contains(re.escape(artist), case=False, na=False)
        mask = mask | artist_mask

    for track in top_affinity_items(profile.get("track_affinity", {}), limit=4):
        track_mask = catalog["track_name"].str.contains(re.escape(track), case=False, na=False)
        mask = mask | track_mask

    genres = set(top_affinity_items(profile.get("genre_affinity", {}), limit=4))
    if genres:
        genre_mask = catalog["track_genre"].str.lower().isin(genres)
        mask = mask | genre_mask

    for liked in profile.get("liked_songs", [])[:12]:
        if liked.get("artists"):
            mask = mask | catalog["artists"].str.contains(re.escape(liked["artists"]), case=False, na=False)
        if liked.get("track_name"):
            mask = mask | catalog["track_name"].str.contains(re.escape(liked["track_name"]), case=False, na=False)
        if liked.get("genre"):
            mask = mask | catalog["track_genre"].str.contains(re.escape(liked["genre"]), case=False, na=False)

    return catalog.loc[mask].copy()


def taste_constraints(profile: dict[str, Any], catalog: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], int]:
    matched = preference_matches(catalog, profile)
    if matched.empty:
        return {}, 0

    constraints: dict[str, tuple[float, float]] = {}
    for feature in ("danceability", "energy", "valence", "acousticness", "instrumentalness"):
        mean = float(matched[feature].mean())
        spread = 0.17
        constraints[feature] = (max(0.0, mean - spread), min(1.0, mean + spread))

    tempo_mean = float(matched["tempo"].mean())
    constraints["tempo_bpm"] = (max(60.0, tempo_mean - 18.0), min(190.0, tempo_mean + 18.0))
    return constraints, len(matched)


def blend_constraint_ranges(
    base: dict[str, tuple[float, float]],
    taste: dict[str, tuple[float, float]],
    strength: float = 0.32,
) -> dict[str, tuple[float, float]]:
    merged = dict(base)
    for feature, (taste_lo, taste_hi) in taste.items():
        if feature not in merged:
            merged[feature] = (taste_lo, taste_hi)
            continue

        base_lo, base_hi = merged[feature]
        base_mid = (base_lo + base_hi) / 2.0
        taste_mid = (taste_lo + taste_hi) / 2.0
        base_span = base_hi - base_lo
        taste_span = taste_hi - taste_lo

        mid = (1.0 - strength) * base_mid + strength * taste_mid
        span = (1.0 - strength) * base_span + strength * taste_span
        lo = mid - span / 2.0
        hi = mid + span / 2.0

        if feature == "tempo_bpm":
            merged[feature] = (max(40.0, lo), min(220.0, hi))
        else:
            merged[feature] = (max(0.0, lo), min(1.0, hi))
    return merged


def apply_taste_profile(spec: QuerySpec, catalog: pd.DataFrame) -> tuple[QuerySpec, str | None]:
    profile = current_taste_profile()
    taste_prior, matched_count = taste_constraints(profile, catalog)
    if not taste_prior:
        return clone_spec(spec), None

    updated = clone_spec(spec)
    updated.constraints = blend_constraint_ranges(updated.constraints, taste_prior)
    if not updated.genres:
        updated.genres = top_affinity_items(profile.get("genre_affinity", {}), limit=3)
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    top_genres = top_affinity_items(profile.get("genre_affinity", {}), limit=2)
    top_artists = top_affinity_items(profile.get("artist_affinity", {}), limit=2)
    parts = []
    if top_genres:
        parts.append("genres: " + ", ".join(top_genres))
    if top_artists:
        parts.append("artists: " + ", ".join(top_artists))
    note = f"I folded in your inferred listening habits from **{matched_count} matched tracks** ({'; '.join(parts) or 'general listening history'})."
    return updated, note


@st.cache_data(ttl=3000, show_spinner=False)
def spotify_access_token(client_id: str, client_secret: str) -> str | None:
    credentials = f"{client_id}:{client_secret}".encode("utf-8")
    encoded = base64.b64encode(credentials).decode("utf-8")
    payload = urllib_parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    request = urllib_request.Request(
        "https://accounts.spotify.com/api/token",
        data=payload,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8")).get("access_token")
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError):
        return None


def spotify_metadata(song) -> dict[str, str | None]:
    cache = current_taste_profile().setdefault("spotify_art_cache", {})
    if song.track_id in cache:
        return cache[song.track_id]

    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret or not song.track_id:
        return {}

    token = spotify_access_token(client_id, client_secret)
    if not token:
        return {}

    request = urllib_request.Request(
        f"https://api.spotify.com/v1/tracks/{song.track_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError):
        return {}

    images = (payload.get("album") or {}).get("images") or []
    metadata = {
        "image_url": images[0]["url"] if images else None,
        "spotify_url": ((payload.get("external_urls") or {}).get("spotify")),
        "preview_url": payload.get("preview_url"),
    }
    cache[song.track_id] = metadata
    save_ui_state(current_taste_profile())
    return metadata


def spotify_api_get(url: str, token: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
    if params:
        url = f"{url}?{urllib_parse.urlencode(params)}"
    request = urllib_request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError):
        return None


def sync_spotify_user_preferences(force: bool = False) -> str | None:
    state = current_taste_profile()
    if state.get("spotify_user_seeded") and not force:
        summary = state.get("spotify_user_summary") or {}
        if summary:
            artists = ", ".join(summary.get("artists", [])[:3])
            genres = ", ".join(summary.get("genres", [])[:3])
            return f"Spotify history loaded: {artists or 'artists unavailable'} | {genres or 'genres unavailable'}."
        return None

    token = os.getenv("SPOTIFY_USER_ACCESS_TOKEN") or os.getenv("SPOTIFY_ACCESS_TOKEN")
    if not token:
        return None

    top_artists = spotify_api_get(
        "https://api.spotify.com/v1/me/top/artists",
        token,
        {"limit": "12", "time_range": "medium_term"},
    )
    top_tracks = spotify_api_get(
        "https://api.spotify.com/v1/me/top/tracks",
        token,
        {"limit": "12", "time_range": "medium_term"},
    )
    if not top_artists and not top_tracks:
        return None

    artist_names: list[str] = []
    genre_names: list[str] = []
    track_names: list[str] = []
    liked_payloads: list[dict[str, Any]] = []

    for artist in (top_artists or {}).get("items", []) or []:
        name = artist.get("name")
        if name:
            artist_names.append(name)
        genre_names.extend(artist.get("genres") or [])

    for track in (top_tracks or {}).get("items", []) or []:
        name = track.get("name")
        if name:
            track_names.append(name)
        liked_payloads.append(
            {
                "track_id": track.get("id", ""),
                "track_name": track.get("name", ""),
                "artists": ";".join(artist.get("name", "") for artist in track.get("artists", []) if artist.get("name")),
                "album_name": (track.get("album") or {}).get("name", ""),
                "genre": "",
            }
        )

    bump_affinity(state.setdefault("artist_affinity", {}), artist_names, 2.2)
    bump_affinity(state.setdefault("genre_affinity", {}), genre_names, 1.8)
    bump_affinity(state.setdefault("track_affinity", {}), track_names, 1.6)

    existing = state.setdefault("liked_songs", [])
    deduped = []
    seen = set()
    for item in liked_payloads + existing:
        key = (item.get("track_id"), item.get("track_name"), item.get("artists"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    state["liked_songs"] = deduped[:24]
    state["spotify_user_seeded"] = True
    state["spotify_user_summary"] = {
        "artists": artist_names[:5],
        "genres": genre_names[:5],
        "tracks": track_names[:5],
    }
    save_ui_state(state)

    artists = ", ".join(artist_names[:3]) or "top artists"
    genres = ", ".join(genre_names[:3]) or "top genres"
    return f"Loaded your Spotify history first: {artists} | {genres}."


def cover_data_uri(title: str, subtitle: str, accent: str) -> str:
    seed = abs(hash(f"{title}|{subtitle}|{accent}"))
    hue = seed % 360
    hue2 = (hue + 55) % 360
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl({hue}, 68%, 56%)"/>
          <stop offset="100%" stop-color="hsl({hue2}, 74%, 28%)"/>
        </linearGradient>
      </defs>
      <rect width="640" height="640" rx="44" fill="url(#g)"/>
      <circle cx="515" cy="124" r="108" fill="rgba(255,255,255,0.12)"/>
      <circle cx="148" cy="492" r="160" fill="rgba(0,0,0,0.10)"/>
      <text x="54" y="420" fill="white" font-size="46" font-weight="700" font-family="Helvetica, Arial, sans-serif">{html.escape(title[:26])}</text>
      <text x="54" y="474" fill="rgba(255,255,255,0.82)" font-size="24" font-family="Helvetica, Arial, sans-serif">{html.escape(subtitle[:36])}</text>
      <text x="54" y="92" fill="rgba(255,255,255,0.90)" font-size="20" letter-spacing="6" font-family="Helvetica, Arial, sans-serif">BAYESIAN DJ</text>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def song_art(song) -> dict[str, str | None]:
    metadata = spotify_metadata(song)
    if metadata.get("image_url"):
        return metadata
    return {
        "image_url": cover_data_uri(song.track_name, song.artists.split(";")[0], song.genre),
        "spotify_url": metadata.get("spotify_url"),
        "preview_url": metadata.get("preview_url"),
    }


def recommendation_queue(session: DJSession, limit: int = 7) -> list[tuple[float, object]]:
    if session.model is None or session.pool.n_available == 0:
        return []

    feat_matrix = session.pool.get_feature_matrix()
    posterior_scores = session.model.predict_proba_posterior(feat_matrix)
    popularity_scores = session.pool.get_popularity_scores()
    scores = 0.80 * posterior_scores + 0.20 * popularity_scores
    indices = session.pool.available_indices()
    queue: list[tuple[float, object]] = []
    current_idx = session._current_song.pool_idx if session._current_song is not None else None

    ordered = np.argsort(scores)[::-1]
    for local_idx in ordered:
        pool_idx = int(indices[local_idx])
        if current_idx is not None and pool_idx == current_idx:
            continue
        queue.append((float(scores[local_idx]), session.pool.get_song_info(pool_idx)))
        if len(queue) >= limit:
            break
    return queue


def render_song_rail(items: list[dict[str, str]], title: str, subtitle: str) -> None:
    if not items:
        return

    cards = []
    for item in items:
        cover = item["image_url"]
        link_open = f'<a href="{item["link"]}" target="_blank" style="text-decoration:none;color:inherit;">' if item.get("link") else ""
        link_close = "</a>" if item.get("link") else ""
        cards.append(
            f"""
            <div class="rail-card">
                {link_open}
                <img class="rail-cover" src="{cover}" alt="{html.escape(item['title'])}" />
                <div class="rail-title">{html.escape(item['title'])}</div>
                <div class="rail-subtitle">{html.escape(item['subtitle'])}</div>
                <div class="rail-meta">{html.escape(item['meta'])}</div>
                {link_close}
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="section-label">{html.escape(title)}</div>
        <div class="rail-shell">
            <div class="rail-header">{html.escape(subtitle)}</div>
            <div class="rail-scroll">
                {''.join(cards)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def favorite_rail_items(catalog: pd.DataFrame) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    profile = current_taste_profile()
    matched = taste_matches(catalog, profile).sort_values("popularity", ascending=False).head(8)
    for _, row in matched.iterrows():
        song_stub = type(
            "SongStub",
            (),
            {
                "track_id": str(row["track_id"]),
                "track_name": str(row["track_name"]),
                "artists": str(row["artists"]),
                "album_name": str(row["album_name"]),
                "genre": str(row["track_genre"]),
            },
        )()
        art = song_art(song_stub)
        items.append(
            {
                "image_url": art["image_url"],
                "title": str(row["track_name"]),
                "subtitle": str(row["artists"]),
                "meta": f"{row['track_genre']} · pop {int(row['popularity'])}",
                "link": art.get("spotify_url") or "",
            }
        )
    return items


def liked_song_rail_items() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for liked in current_taste_profile().get("liked_songs", [])[:10]:
        song_stub = type(
            "SongStub",
            (),
            {
                "track_id": liked.get("track_id", ""),
                "track_name": liked.get("track_name", ""),
                "artists": liked.get("artists", ""),
                "album_name": liked.get("album_name", ""),
                "genre": liked.get("genre", ""),
            },
        )()
        art = song_art(song_stub)
        items.append(
            {
                "image_url": art["image_url"],
                "title": liked.get("track_name", ""),
                "subtitle": liked.get("artists", ""),
                "meta": liked.get("genre", ""),
                "link": art.get("spotify_url") or "",
            }
        )
    return items


def start_session(prompt: str, playlist_length: int) -> None:
    spotify_note = sync_spotify_user_preferences()
    spec = get_parser().parse(prompt)
    infer_preferences_from_message(prompt, spec)
    spec, taste_note = apply_taste_profile(spec, load_catalog())
    session = build_session_from_spec(spec, playlist_length)
    st.session_state["dj_session"] = session
    st.session_state["session_finished"] = session._current_song is None
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["latest_model_update"] = "Session initialized from your opening request."
    add_chat_message("user", prompt)
    response = summarize_spec(spec, session)
    if spotify_note:
        response = f"{spotify_note} {response}"
    if taste_note:
        response = f"{taste_note} {response}"
    add_chat_message("assistant", response)


def chips(items: list[str], tone: str = "") -> str:
    if not items:
        return '<span class="chip">Open-ended</span>'
    class_name = f"chip {tone}".strip()
    return "".join(
        f'<span class="{class_name}">{html.escape(item)}</span>'
        for item in items
    )


def format_constraints(spec) -> list[str]:
    labels: list[str] = []
    for feature, (low, high) in spec.constraints.items():
        nice = AUDIO_FEATURE_LABELS.get(feature.replace("_bpm", ""), feature.replace("_", " ").title())
        if feature == "tempo_bpm":
            labels.append(f"{nice}: {low:.0f}-{high:.0f} bpm")
        else:
            labels.append(f"{nice}: {low:.2f}-{high:.2f}")
    return labels


def remove_negated_tags(items: list[str], message: str) -> list[str]:
    remaining = list(items)
    for item in items:
        pattern = rf"\b(?:no|not|without|less)\s+{re.escape(item)}\b"
        if re.search(pattern, message):
            remaining = [value for value in remaining if value != item]
    return remaining


def merge_specs(current: QuerySpec, refinement: QuerySpec, raw_message: str) -> QuerySpec:
    message = raw_message.lower()
    if any(phrase in message for phrase in ("start over", "new direction", "reset everything", "new vibe")):
        return clone_spec(refinement)

    merged = clone_spec(current)
    merged.genres = remove_negated_tags(merged.genres, message)
    merged.moods = remove_negated_tags(merged.moods, message)

    if any(phrase in message for phrase in ("clear genres", "open genre filter", "any genre", "no genre filter")):
        merged.genres = []
    if any(phrase in message for phrase in ("clear moods", "any mood", "no mood filter", "clear vibes")):
        merged.moods = []
    if any(phrase in message for phrase in ("reset priors", "clear priors", "clear constraints", "forget the priors")):
        merged.constraints = {}

    if refinement.genres:
        merged.genres = list(refinement.genres)
    if refinement.moods:
        merged.moods = list(refinement.moods)
    if refinement.constraints:
        merged.constraints.update(refinement.constraints)
    if refinement.year_range is not None:
        merged.year_range = refinement.year_range
    if refinement.seed_track:
        merged.seed_track = refinement.seed_track
    if refinement.seed_artists:
        merged.seed_artists = list(refinement.seed_artists)

    merged.spotify_search_queries = merged.to_spotify_search_queries()
    return merged


def describe_changes(previous: QuerySpec, updated: QuerySpec) -> str:
    changes: list[str] = []
    if previous.genres != updated.genres:
        changes.append("genres -> " + (", ".join(updated.genres) if updated.genres else "open"))
    if previous.moods != updated.moods:
        changes.append("moods -> " + (", ".join(updated.moods) if updated.moods else "open"))
    changed_constraints = []
    for key, value in updated.constraints.items():
        if previous.constraints.get(key) != value:
            changed_constraints.append(key)
    removed_constraints = [key for key in previous.constraints if key not in updated.constraints]
    if changed_constraints:
        changes.append("priors updated for " + ", ".join(changed_constraints[:3]))
    if removed_constraints:
        changes.append("cleared " + ", ".join(removed_constraints[:3]))
    return "; ".join(changes)


def apply_refinement(message: str) -> None:
    current_session = st.session_state["dj_session"]
    previous_spec = clone_spec(current_session.spec)
    refinement = get_parser().parse(message)
    infer_preferences_from_message(message, refinement)
    spotify_note = sync_spotify_user_preferences()
    merged_spec = merge_specs(previous_spec, refinement, message)
    merged_spec, taste_note = apply_taste_profile(merged_spec, load_catalog())
    prior_feedback = list(zip(current_session.playlist, current_session.actions))
    rebuilt_session = build_session_from_spec(
        merged_spec,
        current_session.playlist_length,
        prior_feedback=prior_feedback,
    )

    if getattr(rebuilt_session, "initial_candidate_count", rebuilt_session.pool.n_available) == 0:
        add_chat_message("user", message)
        add_chat_message(
            "assistant",
            "That refinement collapsed the candidate pool to zero, so I kept the previous priors and posterior intact. Try a broader follow-up.",
        )
        return

    st.session_state["dj_session"] = rebuilt_session
    st.session_state["session_finished"] = rebuilt_session._current_song is None
    st.session_state["last_feedback"] = None
    add_chat_message("user", message)

    change_summary = describe_changes(previous_spec, merged_spec)
    if not change_summary:
        change_summary = "I kept the structure mostly intact and treated your note as a softer steering signal."
    st.session_state["latest_model_update"] = change_summary
    replay_note = f"I replayed **{len(prior_feedback)} feedback decisions** into the updated prior so the posterior kept its memory."
    assistant_reply = f"{change_summary}. {replay_note} {summarize_spec(merged_spec, rebuilt_session)}"
    if spotify_note:
        assistant_reply = f"{spotify_note} {assistant_reply}"
    if taste_note:
        assistant_reply = f"{taste_note} {assistant_reply}"
    add_chat_message("assistant", assistant_reply)


GENRE_TRANSITIONS = {
    "indie": ["indie pop", "alternative", "folk"],
    "hip hop": ["r&b", "neo soul", "house"],
    "trance": ["melodic techno", "progressive house", "dance"],
    "house": ["disco", "dance", "soul"],
    "jazz": ["neo soul", "soul", "funk"],
    "rock": ["indie", "new wave", "alternative"],
}


def next_transition_genre(spec: QuerySpec) -> str | None:
    current = spec.genres[0] if spec.genres else None
    if current:
        options = GENRE_TRANSITIONS.get(current.lower(), [])
        for candidate in options:
            if candidate not in spec.genres:
                return candidate

    preferred = top_affinity_items(current_taste_profile().get("genre_affinity", {}), limit=4)
    for candidate in preferred:
        if candidate not in spec.genres:
            return candidate

    fallback = ["neo soul", "indie pop", "house", "jazz", "folk"]
    for candidate in fallback:
        if candidate not in spec.genres:
            return candidate
    return current


def schedule_speech(text: str, key: str) -> None:
    st.session_state["speech_payload"] = {"text": text, "key": key}


def maybe_trigger_dj_interlude() -> None:
    session = st.session_state["dj_session"]
    rounds = len(session.playlist)
    if rounds == 0 or rounds % 4 != 0:
        return
    if st.session_state.get("last_transition_round") == rounds:
        return

    new_spec = clone_spec(session.spec)
    transition_genre = next_transition_genre(new_spec)
    if transition_genre:
        new_spec.genres = [transition_genre]

    new_spec, _ = apply_taste_profile(new_spec, load_catalog())
    prior_feedback = list(zip(session.playlist, session.actions))
    rebuilt = build_session_from_spec(new_spec, session.playlist_length, prior_feedback=prior_feedback)
    if getattr(rebuilt, "initial_candidate_count", 0) > 0:
        st.session_state["dj_session"] = rebuilt
        st.session_state["session_finished"] = rebuilt._current_song is None

    st.session_state["last_transition_round"] = rounds
    message = (
        f"DJ Bayes here. You've moved through {rounds} songs, so I'm easing the set toward "
        f"{transition_genre or 'a new lane'} while keeping the posterior memory intact."
    )
    add_chat_message("assistant", message)
    st.session_state["latest_model_update"] = f"Auto-transition triggered after {rounds} songs toward {transition_genre or 'a new lane'}."
    schedule_speech(message, f"transition-{rounds}")


def render_voice_interlude() -> None:
    payload = st.session_state.get("speech_payload")
    if not payload:
        return

    components.html(
        f"""
        <script>
        const text = {json.dumps(payload["text"])};
        const key = {json.dumps(payload["key"])};
        const storeKey = "dj-bayes-last-spoken";
        if (window.sessionStorage.getItem(storeKey) !== key) {{
            window.sessionStorage.setItem(storeKey, key);
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.02;
            utterance.lang = "en-US";
            window.speechSynthesis.speak(utterance);
        }}
        </script>
        """,
        height=0,
    )
    st.session_state["speech_payload"] = None


def render_metric_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-subtext">{html.escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_history_frame(session: DJSession) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    snaps = session.model.history[1:] if session.model is not None else []
    for idx, (song, action) in enumerate(zip(session.playlist, session.actions), start=1):
        snap = snaps[idx - 1] if idx - 1 < len(snaps) else None
        rows.append(
            {
                "step": idx,
                "track": song.track_name,
                "artist": song.artists,
                "genre": song.genre,
                "feedback": "Play" if action == "play" else "Skip",
                "posterior": snap.pred_posterior if snap and snap.pred_posterior is not None else None,
                "entropy": snap.entropy if snap else None,
            }
        )
    return pd.DataFrame(rows)


def match_label(score: float) -> str:
    if score >= 0.82:
        return "strong match"
    if score >= 0.64:
        return "good match"
    if score >= 0.46:
        return "developing match"
    return "exploratory pick"


def weight_frame(session: DJSession) -> pd.DataFrame:
    summary = session.model.get_summary()
    frame = (
        pd.DataFrame(
            {
                "feature": list(summary.feature_weights.keys()),
                "weight": list(summary.feature_weights.values()),
            }
        )
        .sort_values("weight")
    )
    frame["direction"] = frame["weight"].apply(lambda x: "Prefers more" if x >= 0 else "Prefers less")
    return frame


def plot_weight_view(session: DJSession):
    frame = weight_frame(session)
    fig = px.bar(
        frame,
        x="weight",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={"Prefers more": "#c95b35", "Prefers less": "#274c43"},
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Posterior weight",
        yaxis_title="",
        legend_title="",
    )
    return fig


def entropy_frame(session: DJSession) -> pd.DataFrame:
    if not session.model.history:
        return pd.DataFrame(columns=["step", "entropy", "entropy_delta"])
    baseline = session.model.history[0].entropy
    frame = pd.DataFrame(
        {
            "step": [snap.step for snap in session.model.history],
            "entropy": [snap.entropy for snap in session.model.history],
        }
    )
    frame["entropy_delta"] = frame["entropy"] - baseline
    return frame


def prior_posterior_frame(session: DJSession) -> pd.DataFrame:
    if not session.model.history:
        return pd.DataFrame(columns=["feature", "Prior", "Posterior", "Shift"])

    initial = session.model.history[0].mu
    current = session.model.mu
    rows = []
    for idx, feature in enumerate(AUDIO_FEATURES, start=1):
        prior = float(initial[idx])
        posterior = float(current[idx])
        rows.append(
            {
                "feature": AUDIO_FEATURE_LABELS.get(feature, feature.title()),
                "Prior": prior,
                "Posterior": posterior,
                "Shift": posterior - prior,
            }
        )
    return pd.DataFrame(rows)


def plot_entropy_view(session: DJSession):
    frame = entropy_frame(session)
    fig = px.line(frame, x="step", y="entropy_delta", markers=True)
    fig.update_traces(line_color="#274c43", marker_color="#c95b35", line_width=3)
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Feedback round",
        yaxis_title="Entropy change vs prior",
    )
    return fig


def plot_prior_posterior_view(session: DJSession):
    frame = prior_posterior_frame(session)
    melted = frame.melt(id_vars="feature", value_vars=["Prior", "Posterior"], var_name="state", value_name="weight")
    fig = px.bar(
        melted,
        x="feature",
        y="weight",
        color="state",
        barmode="group",
        color_discrete_map={"Prior": "#9bb8ff", "Posterior": "#1ed760"},
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Weight",
        legend_title="",
    )
    return fig


def plot_shift_view(session: DJSession):
    frame = prior_posterior_frame(session)
    fig = px.bar(
        frame,
        x="feature",
        y="Shift",
        color="Shift",
        color_continuous_scale=["#ff8e6e", "#f5f9ff", "#1ed760"],
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Posterior shift",
        coloraxis_showscale=False,
    )
    fig.add_hline(y=0.0, line_width=1, line_color="rgba(255,255,255,0.35)")
    return fig


def feature_profile_frame(song) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": [AUDIO_FEATURE_LABELS.get(name, name.title()) for name in song.features],
            "value": list(song.features.values()),
        }
    )
    return frame


def plot_feature_profile(song):
    frame = feature_profile_frame(song)
    fig = px.bar(
        frame,
        x="feature",
        y="value",
        color="value",
        color_continuous_scale=["#f0b38f", "#c95b35", "#274c43"],
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis_title="Normalized audio feature",
        coloraxis_showscale=False,
    )
    return fig


def render_weight_chart(session: DJSession) -> None:
    frame = weight_frame(session)
    if HAS_PLOTLY:
        st.plotly_chart(plot_weight_view(session), width="stretch")
        return

    fallback = frame.set_index("feature")[["weight"]]
    st.bar_chart(fallback, width="stretch")
    st.caption("Install `plotly` for the full styled posterior weight chart.")


def render_entropy_chart(session: DJSession) -> None:
    frame = entropy_frame(session).set_index("step")
    if HAS_PLOTLY:
        st.plotly_chart(plot_entropy_view(session), width="stretch")
        return

    st.line_chart(frame[["entropy_delta"]], width="stretch")
    st.caption("Entropy change is measured relative to the initial prior state.")


def render_prior_posterior_chart(session: DJSession) -> None:
    frame = prior_posterior_frame(session).set_index("feature")
    if HAS_PLOTLY:
        st.plotly_chart(plot_prior_posterior_view(session), width="stretch")
        return

    st.bar_chart(frame[["Prior", "Posterior"]], width="stretch")


def render_shift_chart(session: DJSession) -> None:
    frame = prior_posterior_frame(session).set_index("feature")
    if HAS_PLOTLY:
        st.plotly_chart(plot_shift_view(session), width="stretch")
        return

    st.bar_chart(frame[["Shift"]], width="stretch")


def render_feature_chart(song) -> None:
    frame = feature_profile_frame(song)
    if HAS_PLOTLY:
        st.plotly_chart(plot_feature_profile(song), width="stretch")
        return

    st.bar_chart(frame.set_index("feature")[["value"]], width="stretch")
    st.caption("Install `plotly` for the full styled audio feature chart.")


def render_hero(catalog_size: int) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Bayesian Music Copilot</div>
            <div class="hero-title">A listening interface that learns in public.</div>
            <p class="hero-copy">
                Describe a mood like you would prompt an LLM. Bayesian DJ parses the request,
                turns it into priors over audio features, then updates its posterior every time
                you play or skip a recommendation across {catalog_size:,} tracks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prompt_preset_row() -> None:
    st.markdown('<div class="section-label">Quick Starts</div>', unsafe_allow_html=True)
    preset_columns = st.columns(len(PROMPT_PRESETS))
    for idx, (column, prompt) in enumerate(zip(preset_columns, PROMPT_PRESETS), start=1):
        with column:
            if st.button(prompt, key=f"preset_{idx}", width="stretch"):
                st.session_state["prompt_input"] = prompt


def render_conversation() -> None:
    st.markdown('<div class="section-label">Chat with the DJ</div>', unsafe_allow_html=True)
    for message in st.session_state["chat_messages"][-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_refinement_presets() -> None:
    return None


def render_taste_profile_controls(catalog: pd.DataFrame) -> None:
    return None


def preference_summary_text() -> str:
    state = current_taste_profile()
    genres = top_affinity_items(state.get("genre_affinity", {}), limit=3)
    artists = top_affinity_items(state.get("artist_affinity", {}), limit=3)
    liked_count = len(state.get("liked_songs", []))
    parts = []
    if genres:
        parts.append("genres: " + ", ".join(genres))
    if artists:
        parts.append("artists: " + ", ".join(artists))
    if liked_count:
        parts.append(f"liked songs remembered: {liked_count}")
    spotify_summary = state.get("spotify_user_summary") or {}
    if spotify_summary.get("artists"):
        parts.append("spotify top artists: " + ", ".join(spotify_summary["artists"][:2]))
    return " | ".join(parts) if parts else "No stable listening profile inferred yet. Tell DJ Bayes what you usually listen to."


def render_latest_update() -> None:
    update = st.session_state.get("latest_model_update") or "No posterior update yet."
    preference_text = preference_summary_text()
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Latest Model Update</div>
            <div class="playlist-title">{html.escape(update)}</div>
            <div class="playlist-meta" style="margin-top:0.55rem;">Inferred listening profile: {html.escape(preference_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(message: str, tone: str = "info") -> None:
    st.markdown(
        f'<div class="status-banner {html.escape(tone)}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def spotify_links(song, art: dict[str, str | None]) -> tuple[str | None, str | None]:
    track_id = getattr(song, "track_id", "")
    spotify_web_url = art.get("spotify_url") or (
        f"https://open.spotify.com/track/{urllib_parse.quote(track_id)}" if track_id else None
    )
    spotify_app_url = f"spotify:track:{urllib_parse.quote(track_id)}" if track_id else None
    return spotify_app_url, spotify_web_url


def open_spotify_track(song, art: dict[str, str | None]) -> bool:
    spotify_app_url, spotify_web_url = spotify_links(song, art)
    for url in (spotify_app_url, spotify_web_url):
        if not url:
            continue
        try:
            if webbrowser.open(url, new=0, autoraise=True):
                return True
        except Exception:
            continue
    return False


def render_playback_area(song, art: dict[str, str | None]) -> None:
    if not st.session_state.get("is_playing"):
        return

    preview_url = art.get("preview_url")
    spotify_app_url, spotify_web_url = spotify_links(song, art)
    track_id = getattr(song, "track_id", "")
    if preview_url:
        audio_markup = f"""
        <div class="playback-shell">
            <div class="playback-title">Now Playing Preview</div>
            <audio id="bayes-player" src="{html.escape(preview_url)}" controls autoplay style="width:100%;"></audio>
            <div id="bayes-player-status" class="playback-meta">Starting the Spotify preview.</div>
            <script>
                const audio = document.getElementById("bayes-player");
                const status = document.getElementById("bayes-player-status");
                async function startPlayback() {{
                    try {{
                        const maybePromise = audio.play();
                        if (maybePromise) {{
                            await maybePromise;
                        }}
                        status.textContent = "Preview playing now.";
                    }} catch (err) {{
                        status.textContent = "Autoplay was blocked by the browser. Use the player controls above to start playback.";
                    }}
                }}
                startPlayback();
                audio.addEventListener("play", () => {{
                    status.textContent = "Preview playing now.";
                }});
                audio.addEventListener("pause", () => {{
                    status.textContent = "Preview paused.";
                }});
            </script>
        </div>
        """
        components.html(audio_markup, height=138)
        return

    if track_id:
        embed_url = f"https://open.spotify.com/embed/track/{urllib_parse.quote(track_id)}?utm_source=generator"
        spotify_link = html.escape(spotify_web_url or "")
        spotify_app_link = html.escape(spotify_app_url or "")
        embed_markup = f"""
        <div class="playback-shell">
            <div class="playback-title">Spotify Playback</div>
            <iframe
                src="{embed_url}"
                width="100%"
                height="152"
                frameborder="0"
                allowfullscreen=""
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"></iframe>
            <div class="playback-meta">
                This track does not expose a preview clip. DJ Bayes tried to open Spotify directly.
                <a href="{spotify_app_link}" target="_top" style="color:#88f0b2;">Open in Spotify app</a>
                or
                <a href="{spotify_link}" target="_blank" style="color:#88f0b2;">open it in Spotify Web</a>.
            </div>
        </div>
        """
        components.html(embed_markup, height=248)
        return

    render_status_banner(
        "Playback is enabled, but Spotify does not provide a preview or embed for this track.",
        tone="warning",
    )


def render_sidebar(catalog: pd.DataFrame) -> int:
    return st.session_state.get("playlist_length", 12)


def render_current_track(session: DJSession) -> None:
    ensure_current_song(session)
    playback_song = deserialize_song(st.session_state.get("playback_song"))
    song = playback_song or session._current_song
    if song is None:
        render_status_banner("No current recommendation is available.", tone="info")
        return

    art = song_art(song)
    if playback_song is not None:
        x = np.array([1.0, *[song.features[name] for name in AUDIO_FEATURES]], dtype=np.float64).reshape(1, -1)
    else:
        x = session._current_x.reshape(1, -1)
    posterior_score = float(session.model.predict_proba_posterior(x)[0])
    label = match_label(posterior_score)

    strip_cols = st.columns([0.18, 0.82], vertical_alignment="center")
    with strip_cols[0]:
        st.image(art["image_url"], width="stretch")
    with strip_cols[1]:
        st.markdown(
            f"""
            <div class="deck-shell">
                <div class="track-kicker">DJ Bayes</div>
                <div class="track-title">{html.escape(song.track_name)}</div>
                <div class="track-meta">{html.escape(song.artists)} · {html.escape(song.album_name)} · {label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_playback_area(song, art)

    if st.session_state.get("session_finished", False):
        render_status_banner(
            "Session closed. Reset from the sidebar or start a new prompt to keep training the model.",
            tone="info",
        )
    else:
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("Play", width="stretch", type="primary"):
                st.session_state["is_playing"] = True
                launched_in_spotify = open_spotify_track(song, art)
                if playback_song is None and not st.session_state.get("playback_scored", False):
                    st.session_state["playback_song"] = serialize_song(song)
                    deltas = session.record_feedback(True)
                    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
                    infer_preferences_from_song(song, played=True)
                    st.session_state["playback_scored"] = True
                    launch_prefix = "Opened in Spotify. " if launched_in_spotify else ""
                    st.session_state["last_feedback"] = f"{launch_prefix}Playing now. Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
                    st.session_state["latest_model_update"] = f"You played {song.track_name}. The posterior moved most on {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."
                    add_song_to_liked(song)
                    if len(session.playlist) >= session.playlist_length:
                        st.session_state["session_finished"] = True
                    else:
                        ensure_current_song(session)
                    maybe_trigger_dj_interlude()
                elif launched_in_spotify:
                    st.session_state["last_feedback"] = f"Opened {song.track_name} in Spotify."
                st.rerun()
        with action_cols[1]:
            if st.button("Pause", width="stretch"):
                st.session_state["is_playing"] = False
                st.rerun()
        with action_cols[2]:
            if st.button("Skip", width="stretch"):
                st.session_state["is_playing"] = False
                if playback_song is not None:
                    st.session_state["playback_song"] = None
                    st.session_state["playback_scored"] = False
                    if session._current_song is None and not session_complete(session):
                        ensure_current_song(session)
                    st.session_state["last_feedback"] = f"Skipped ahead from {song.track_name}."
                else:
                    deltas = session.record_feedback(False)
                    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
                    infer_preferences_from_song(song, played=False)
                    st.session_state["last_feedback"] = f"Skipped. Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
                    st.session_state["latest_model_update"] = f"You skipped {song.track_name}. The posterior pushed away most from {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."
                    if len(session.playlist) >= session.playlist_length:
                        st.session_state["session_finished"] = True
                    else:
                        ensure_current_song(session)
                    maybe_trigger_dj_interlude()
                st.rerun()

    if st.session_state.get("last_feedback"):
        render_status_banner(st.session_state["last_feedback"], tone="success")


def render_session_overview(session: DJSession) -> None:
    initial_pool = getattr(session, "initial_candidate_count", session.pool.n_available + len(session.playlist))
    liked_count = sum(action == "play" for action in session.actions)
    skipped_count = sum(action == "skip" for action in session.actions)
    entropy = session.model.history[-1].entropy if session.model.history else 0.0

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Filtered pool", f"{initial_pool:,}", "Tracks matched the prompt-derived priors before feedback trimmed the set.")
    with metric_cols[1]:
        render_metric_card("Played / skipped", f"{liked_count} / {skipped_count}", "Feedback is streamed directly into the posterior.")
    with metric_cols[2]:
        render_metric_card("Posterior entropy", f"{entropy:.2f}", "Lower values mean the model has become more certain.")

    spec = session.spec
    st.markdown('<div class="section-label">Prompt Decomposition</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Genres</div>
            <div class="chip-row">{chips(spec.genres, "warm")}</div>
            <div class="metric-label" style="margin-top: 1rem;">Moods</div>
            <div class="chip-row">{chips(spec.moods, "gold")}</div>
            <div class="metric-label" style="margin-top: 1rem;">Feature priors</div>
            <div class="chip-row">{chips(format_constraints(spec))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_posterior_panels(session: DJSession) -> None:
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.markdown('<div class="section-label">Prior vs Posterior</div>', unsafe_allow_html=True)
        render_prior_posterior_chart(session)
    with chart_cols[1]:
        st.markdown('<div class="section-label">Posterior Shift</div>', unsafe_allow_html=True)
        render_shift_chart(session)
    with chart_cols[2]:
        st.markdown('<div class="section-label">Posterior Entropy</div>', unsafe_allow_html=True)
        render_entropy_chart(session)


def render_playlist_column(session: DJSession) -> None:
    summary = session.model.get_summary()
    positive_html = "".join(
        f'<div class="playlist-item"><div class="playlist-title">{html.escape(AUDIO_FEATURE_LABELS.get(name, name.title()))}</div><div class="playlist-meta">Posterior weight {value:+.3f}</div></div>'
        for name, value in summary.top_positive
    ) or '<div class="empty-state">No positive preference shifts yet.</div>'
    negative_html = "".join(
        f'<div class="playlist-item"><div class="playlist-title">{html.escape(AUDIO_FEATURE_LABELS.get(name, name.title()))}</div><div class="playlist-meta">Posterior weight {value:+.3f}</div></div>'
        for name, value in reversed(summary.top_negative)
    ) or '<div class="empty-state">No negative preference shifts yet.</div>'
    liked = [song for song, action in zip(session.playlist, session.actions) if action == "play"]
    liked_html = "".join(
        f'<div class="playlist-item"><div class="playlist-title">{idx}. {html.escape(song.track_name)}</div><div class="playlist-meta">{html.escape(song.artists)} · {html.escape(song.genre)}</div></div>'
        for idx, song in enumerate(liked, start=1)
    ) or '<div class="empty-state">Your final playlist will start filling in after the first positive signal.</div>'

    st.markdown('<div class="section-label">What the model is learning</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Strongest pulls</div>
            {positive_html}
            <div class="metric-label" style="margin-top: 1rem;">Strongest pushbacks</div>
            {negative_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Accepted songs</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glass-card">{liked_html}</div>', unsafe_allow_html=True)


def render_history(session: DJSession) -> None:
    history = build_history_frame(session)
    st.markdown('<div class="section-label">Feedback Log</div>', unsafe_allow_html=True)
    if history.empty:
        st.markdown(
            '<div class="glass-card"><div class="empty-state">No feedback yet. The first play/skip decision will create the posterior trajectory.</div></div>',
            unsafe_allow_html=True,
        )
        return
    st.dataframe(
        history,
        width="stretch",
        hide_index=True,
        column_config={
            "step": st.column_config.NumberColumn("Round"),
            "track": "Track",
            "artist": "Artist",
            "genre": "Genre",
            "feedback": "Decision",
            "posterior": st.column_config.ProgressColumn(
                "Posterior keep prob.",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "entropy": st.column_config.NumberColumn("Entropy", format="%.3f"),
        },
    )


def render_empty_workspace() -> None:
    st.markdown('<div class="section-label">DJ Bayes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            Tell DJ Bayes what you want to hear, and mention what you usually listen to.
            I will infer your music preferences from the conversation and use them to shape the prior automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    init_state()
    session = st.session_state["dj_session"]
    if session is None:
        render_empty_workspace()
    else:
        if session.pool.n_available == 0 and not session.playlist:
            st.error("The prompt filtered the catalog down to zero candidates. Broaden the mood or genre description and try again.")
        else:
            render_current_track(session)
            render_conversation()
            render_posterior_panels(session)
            render_latest_update()

    follow_up = st.chat_input("Tell DJ Bayes what you want, or what you already listen to")
    if follow_up:
        with st.spinner("Updating DJ Bayes..."):
            if st.session_state["dj_session"] is None:
                start_session(follow_up, st.session_state["playlist_length"])
            else:
                apply_refinement(follow_up)
        st.rerun()

    render_voice_interlude()


if __name__ == "__main__":
    main()
