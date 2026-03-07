from __future__ import annotations

import base64
import difflib
import html
import json
import os
import random
import re
import secrets
import time
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
from music_query_parser.parser import KNOWN_MOODS, MOOD_ALIASES, MusicQueryParser, QuerySpec

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
SPOTIFY_OAUTH_SCOPES = "user-top-read user-library-read user-library-modify user-read-recently-played"

DATA_DIR.mkdir(exist_ok=True)

US_HIT_GENRE_TOKENS = (
    "pop",
    "hip hop",
    "rap",
    "r&b",
    "soul",
    "indie",
    "alternative",
    "rock",
    "house",
    "dance",
    "edm",
    "country",
    "folk",
    "jazz",
    "funk",
    "blues",
    "punk",
    "metal",
    "americana",
    "new wave",
    "electronic",
    "disco",
)

DEPRIORITIZED_WORLD_GENRE_TOKENS = (
    "sertanejo",
    "forro",
    "pagode",
    "mpb",
    "samba",
    "brazil",
    "latin",
    "latino",
    "reggaeton",
    "k-pop",
    "j-pop",
    "mandopop",
    "cantopop",
    "bollywood",
    "desi",
    "hindi",
    "tamil",
    "telugu",
    "arab",
    "turkish",
    "french",
    "german",
    "spanish",
    "portuguese",
)

ARTIST_NAME_ALIASES = {
    "pharell": "Pharrell Williams",
    "pharrell": "Pharrell Williams",
    "tyler the creator": "Tyler, The Creator",
    "tyler the creator.": "Tyler, The Creator",
    "weeknd": "The Weeknd",
    "asap rocky": "A$AP Rocky",
    "a$ap rocky": "A$AP Rocky",
    "kendrick": "Kendrick Lamar",
    "sza": "SZA",
    "rihanna": "Rihanna",
    "beyonce": "Beyonce",
    "beyoncé": "Beyonce",
    "ye": "Kanye West",
    "kanye": "Kanye West",
    "drake": "Drake",
    "future": "Future",
    "eminem": "Eminem",
    "21 savage": "21 Savage",
    "chief keef": "Chief Keef",
    "king von": "King Von",
    "lil yachty": "Lil Yachty",
}


@st.cache_data
def load_catalog() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_CSV)


@st.cache_resource
def get_parser() -> MusicQueryParser:
    return MusicQueryParser()


def normalize_artist_name(value: str) -> str:
    normalized = normalize_affinity_label(value)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@st.cache_data
def catalog_artist_lookup() -> tuple[list[str], dict[str, str]]:
    catalog = load_catalog()
    exploded = (
        catalog.assign(artist_name=catalog["artists"].fillna("").str.split(";"))
        .explode("artist_name")
        .assign(artist_name=lambda df: df["artist_name"].fillna("").str.strip())
    )
    exploded = exploded.loc[exploded["artist_name"].ne("")]
    summary = (
        exploded.groupby("artist_name", as_index=False)
        .agg(popularity=("popularity", "mean"), appearances=("track_name", "count"))
        .sort_values(["appearances", "popularity", "artist_name"], ascending=[False, False, True])
    )
    mapping: dict[str, str] = {}
    ordered: list[str] = []
    for artist in summary["artist_name"].tolist():
        normalized = normalize_artist_name(artist)
        if len(normalized) < 3 or normalized in mapping:
            continue
        mapping[normalized] = artist
        ordered.append(normalized)
        if len(ordered) >= 3000:
            break
    for alias, canonical in ARTIST_NAME_ALIASES.items():
        mapping.setdefault(normalize_artist_name(alias), canonical)
        ordered.insert(0, normalize_artist_name(alias))
    return ordered, mapping


def default_ui_state() -> dict[str, Any]:
    return {
        "artist_affinity": {},
        "genre_affinity": {},
        "track_affinity": {},
        "liked_songs": [],
        "spotify_art_cache": {},
        "spotify_user_seeded": False,
        "spotify_user_summary": {},
        "spotify_saved_track_ids": [],
        "spotify_synced_track_ids": [],
        "spotify_auth": {},
        "spotify_oauth_state": "",
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


def spotify_client_credentials() -> tuple[str | None, str | None]:
    return os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET")


def spotify_redirect_uri() -> str:
    return os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501")


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

        .chat-transcript {
            display: grid;
            gap: 0.9rem;
        }

        .chat-bubble {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 12px 30px rgba(7, 22, 63, 0.10);
        }

        .chat-bubble.user {
            background: rgba(255, 255, 255, 0.09);
        }

        .chat-bubble.assistant {
            background: rgba(255, 255, 255, 0.13);
        }

        .chat-role {
            color: var(--accent-soft);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .chat-body {
            color: var(--ink);
            font-size: 1rem;
            line-height: 1.55;
        }

        .chat-body p {
            margin: 0.2rem 0;
        }

        .auth-shell {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid var(--border);
            border-radius: 30px;
            padding: 1.4rem;
            box-shadow: 0 18px 40px rgba(7, 22, 63, 0.14);
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
    if st.session_state.get("_fresh_init_done"):
        return
    st.session_state["dj_session"] = None
    st.session_state["session_finished"] = False
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["prompt_input"] = PROMPT_PRESETS[0]
    st.session_state["ui_state"] = default_ui_state()
    st.session_state["playlist_length"] = 12
    st.session_state["is_playing"] = False
    st.session_state["speech_payload"] = None
    st.session_state["last_transition_round"] = 0
    st.session_state["recent_intervention_routes"] = []
    st.session_state["latest_model_update"] = ""
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    st.session_state["seen_track_ids"] = set()
    st.session_state["pending_reaction"] = None
    st.session_state["spotify_auth_session"] = {}
    st.session_state["spotify_oauth_state_session"] = ""
    st.session_state["_fresh_init_done"] = True


def reset_session() -> None:
    st.session_state["dj_session"] = None
    st.session_state["session_finished"] = False
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["is_playing"] = False
    st.session_state["speech_payload"] = None
    st.session_state["last_transition_round"] = 0
    st.session_state["recent_intervention_routes"] = []
    st.session_state["latest_model_update"] = ""
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    st.session_state["seen_track_ids"] = set()
    st.session_state["pending_reaction"] = None


def session_complete(session: DJSession | None) -> bool:
    if session is None:
        return True
    if st.session_state.get("session_finished", False):
        return True
    if session.pool.n_available == 0 and session._current_song is None:
        return True
    return False


def ensure_current_song(session: DJSession) -> None:
    if session is None:
        return
    if session._current_song is None:
        song = session.recommend_next()
        if song is None:
            st.session_state["session_finished"] = True
        elif getattr(song, "track_id", ""):
            seen = set(st.session_state.get("seen_track_ids", set()))
            seen.add(str(song.track_id))
            st.session_state["seen_track_ids"] = seen


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
    excluded_track_ids: set[str] | None = None,
) -> DJSession:
    catalog = load_catalog()
    session = DJSession(
        csv_path=DEFAULT_CSV,
        playlist_length=playlist_length,
        parser=get_parser(),
    )
    session.spec = clone_spec(spec)
    session.pool.filter_by_genres(session.spec.genres)
    session.pool.mark_used_track_ids(set(excluded_track_ids or set()))
    session.pool.set_external_bias(catalog_preference_scores(catalog, current_taste_profile(), spec).to_numpy(dtype=float))
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

    if session.pool.n_available > 0:
        if not prior_feedback and session.spec.seed_artists:
            session.recommend_next(
                preferred_artists=session.spec.seed_artists,
                require_artist_match=True,
            )
        else:
            session.recommend_next()
    return session


def add_chat_message(role: str, content: str) -> None:
    st.session_state["chat_messages"].append({"role": role, "content": content})


def compose_assistant_message(*sections: str | None) -> str:
    cleaned = [section.strip() for section in sections if section and section.strip()]
    return "\n\n".join(cleaned)


def humanize_spotify_note(note: str | None) -> str | None:
    if not note:
        return None
    lowered = note.lower()
    if lowered.startswith("loaded your spotify history first:"):
        return "**I pulled in your Spotify taste first:** " + note.split(":", 1)[1].strip()
    if lowered.startswith("spotify history loaded:"):
        return "**I'm leaning on your Spotify history here:** " + note.split(":", 1)[1].strip()
    return note


def humanize_taste_note(note: str | None) -> str | None:
    if not note:
        return None
    return "**What I'm keeping in mind:** " + re.sub(r"^I\s+", "", note, count=1)


def render_message_html(content: str) -> str:
    escaped = html.escape(str(content))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = escaped.replace("\n", "<br>")
    return escaped


def summarize_spec(spec: QuerySpec, session: DJSession) -> str:
    descriptors: list[str] = []
    if spec.seed_artists:
        descriptors.append(f"around {', '.join(spec.seed_artists[:2])}")
    if spec.seed_track:
        descriptors.append(f"borrowing some of the feel of {spec.seed_track}")
    if spec.moods:
        descriptors.append(f"in a {', '.join(spec.moods[:2])} pocket")
    elif spec.genres:
        descriptors.append(f"in a {', '.join(spec.genres[:2])} lane")

    if descriptors:
        lines = [f"**Set direction:** I'm starting {', '.join(descriptors)}."]
    else:
        lines = ["**Set direction:** I'm keeping this open and feeling for the best entry point."]
    if session._current_song is not None:
        lines.append(
            f"**Starting with:** {session._current_song.track_name} by {session._current_song.artists}"
        )
    if session.last_recommendation_score is not None:
        lines.append(f"**Read on the room:** {match_label(session.last_recommendation_score)}")
    lines.append("**Talk to me while this plays:** ask for warmer, darker, bigger hooks, less energy, or closer to a different artist.")
    return "\n\n".join(lines)


def parse_preference_text(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r",|\n", raw) if item.strip()]


def current_taste_profile() -> dict[str, Any]:
    return st.session_state["ui_state"]


def normalize_affinity_label(value: str) -> str:
    normalized = value.strip().lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def bump_affinity(bucket: dict[str, float], values: list[str], amount: float) -> None:
    for value in values:
        key = normalize_affinity_label(value)
        if not key:
            continue
        bucket[key] = bucket.get(key, 0.0) + amount


def explicit_prompt_moods(prompt: str) -> list[str]:
    lowered = normalize_affinity_label(prompt)
    moods: list[str] = []
    for mood in KNOWN_MOODS:
        if re.search(rf"\b{re.escape(mood)}\b", lowered):
            moods.append(mood)
    for phrase, mapped in MOOD_ALIASES.items():
        if phrase in lowered:
            moods.extend(mapped)
    deduped: list[str] = []
    seen: set[str] = set()
    for mood in moods:
        if mood in seen:
            continue
        seen.add(mood)
        deduped.append(mood)
    return deduped


def prompt_artist_candidates(prompt: str) -> list[str]:
    normalized_prompt = normalize_artist_name(prompt)
    if not normalized_prompt:
        return []

    artist_keys, mapping = catalog_artist_lookup()
    matches: list[str] = []
    seen: set[str] = set()

    for alias, canonical in ARTIST_NAME_ALIASES.items():
        alias_key = normalize_artist_name(alias)
        if re.search(rf"\b{re.escape(alias_key)}\b", normalized_prompt):
            if canonical not in seen:
                seen.add(canonical)
                matches.append(canonical)

    cue_heavy = bool(
        re.search(
            r"\b(?:like|similar to|based on|based|inspired by|style|vibe|channel|around|with)\b",
            normalized_prompt,
        )
    )

    for key in artist_keys[:1800]:
        if len(matches) >= 3:
            break
        if len(key) < 4:
            continue
        if re.search(rf"\b{re.escape(key)}\b", normalized_prompt):
            canonical = mapping[key]
            if canonical not in seen:
                seen.add(canonical)
                matches.append(canonical)

    if matches or not cue_heavy:
        return matches[:3]

    tokens = [token for token in normalized_prompt.split() if len(token) > 2]
    ngrams: set[str] = set()
    for size in range(1, min(4, len(tokens)) + 1):
        for start in range(0, len(tokens) - size + 1):
            chunk = " ".join(tokens[start:start + size])
            if len(chunk) >= 4:
                ngrams.add(chunk)

    for chunk in sorted(ngrams, key=len, reverse=True):
        close = difflib.get_close_matches(chunk, artist_keys, n=1, cutoff=0.88 if " " in chunk else 0.94)
        if not close:
            continue
        canonical = mapping[close[0]]
        if canonical in seen:
            continue
        seen.add(canonical)
        matches.append(canonical)
        if len(matches) >= 3:
            break

    return matches[:3]


def enrich_spec_from_prompt(prompt: str, spec: QuerySpec) -> QuerySpec:
    enriched = clone_spec(spec)
    artist_mentions = prompt_artist_candidates(prompt)
    if artist_mentions:
        combined_artists = list(dict.fromkeys(artist_mentions + enriched.seed_artists))
        enriched.seed_artists = combined_artists[:3]

    explicit_moods = explicit_prompt_moods(prompt)
    if enriched.seed_artists and explicit_moods:
        enriched.moods = explicit_moods[:4]
    elif explicit_moods:
        retained = [mood for mood in enriched.moods if mood in explicit_moods]
        enriched.moods = list(dict.fromkeys(explicit_moods + retained))[:5]
    else:
        enriched.moods = enriched.moods[:4]

    enriched.spotify_search_queries = enriched.to_spotify_search_queries()
    return enriched


def infer_preferences_from_message(message: str, spec: QuerySpec) -> None:
    lowered = message.lower()
    if not any(
        phrase in lowered
        for phrase in ("usually listen to", "i listen to", "favorite", "love", "mostly listen to")
    ):
        return
    state = current_taste_profile()
    boost = 2.2
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
        state["liked_songs"] = liked[:48]
        save_ui_state(state)


def mark_spotify_track_saved(song) -> None:
    state = current_taste_profile()
    saved_ids = set(state.get("spotify_saved_track_ids", []))
    if song.track_id:
        saved_ids.add(str(song.track_id))
        state["spotify_saved_track_ids"] = sorted(saved_ids)
    save_ui_state(state)


def apply_positive_feedback(session: DJSession, song, source: str) -> None:
    st.session_state["playback_song"] = serialize_song(song)
    st.session_state["playback_scored"] = True
    st.session_state["showing_back_song"] = False
    deltas = session.record_feedback(True)
    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
    infer_preferences_from_song(song, played=True)
    add_song_to_liked(song)
    ensure_current_song(session)
    maybe_trigger_dj_interlude()
    st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
    st.session_state["latest_model_update"] = f"{source.rstrip('.')} The posterior moved most on {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."


def apply_negative_feedback(session: DJSession, song, source: str) -> None:
    deltas = session.record_feedback(False)
    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
    infer_preferences_from_song(song, played=False)
    ensure_current_song(session)
    maybe_trigger_dj_interlude()
    st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
    st.session_state["latest_model_update"] = f"{source.rstrip('.')} The posterior pushed away most from {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."


def maybe_sync_spotify_saved_feedback(session: DJSession, song) -> bool:
    if st.session_state.get("playback_song") is not None:
        return False
    if st.session_state.get("showing_back_song", False):
        return False
    track_id = getattr(song, "track_id", "")
    if not track_id:
        return False
    state = current_taste_profile()
    synced = set(state.get("spotify_synced_track_ids", []))
    if track_id in synced:
        return False
    saved = spotify_track_saved(song)
    if not saved:
        return False
    mark_spotify_track_saved(song)
    apply_positive_feedback(session, song, "Saved on Spotify.")
    synced.add(track_id)
    state["spotify_synced_track_ids"] = sorted(synced)
    save_ui_state(state)
    return True


def queue_pending_reaction(song, source: str, liked: bool) -> None:
    st.session_state["pending_reaction"] = {
        "track_id": str(getattr(song, "track_id", "") or ""),
        "source": source,
        "liked": liked,
    }


def clear_pending_reaction() -> None:
    st.session_state["pending_reaction"] = None


def apply_pending_reaction_if_ready(session: DJSession, song) -> bool:
    pending = st.session_state.get("pending_reaction")
    if not pending:
        return False
    if str(pending.get("track_id", "")) != str(getattr(song, "track_id", "") or ""):
        return False

    source = str(pending.get("source", "Queued feedback."))
    liked = bool(pending.get("liked", False))
    clear_pending_reaction()
    if liked:
        deltas = session.record_feedback(True)
        infer_preferences_from_song(song, played=True)
        add_song_to_liked(song)
        ensure_current_song(session)
        maybe_trigger_dj_interlude()
        top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
        st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
        st.session_state["latest_model_update"] = f"{source.rstrip('.')} The posterior moved most on {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."
        return True

    deltas = session.record_feedback(False)
    infer_preferences_from_song(song, played=False)
    ensure_current_song(session)
    maybe_trigger_dj_interlude()
    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
    st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
    st.session_state["latest_model_update"] = f"{source.rstrip('.')} The posterior pushed away most from {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."
    return True


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


def prompt_has_clear_direction(spec: QuerySpec | None) -> bool:
    if spec is None:
        return False
    return bool(spec.genres or spec.moods or spec.constraints or spec.seed_artists or spec.seed_track)


def catalog_preference_scores(catalog: pd.DataFrame, profile: dict[str, Any], spec: QuerySpec | None = None) -> pd.Series:
    if catalog.empty:
        return pd.Series(dtype=float)

    directed_prompt = prompt_has_clear_direction(spec)
    artist_scale = 1.15 if directed_prompt else 1.6
    track_scale = 0.95 if directed_prompt else 1.25
    genre_scale = 0.28 if directed_prompt else 1.15
    popularity_scale = 2.0 if directed_prompt else 1.6

    score = pd.Series(0.0, index=catalog.index, dtype=float)
    artist_col = catalog["artists"].fillna("")
    track_col = catalog["track_name"].fillna("")
    genre_col = catalog["track_genre"].fillna("").str.lower()
    popularity_rank = catalog["popularity"].rank(method="average", pct=True).astype(float)

    top_artists = sorted(profile.get("artist_affinity", {}).items(), key=lambda item: item[1], reverse=True)[:12]
    for artist, weight in top_artists:
        if weight <= 0:
            continue
        score += artist_col.str.contains(re.escape(artist), case=False, na=False).astype(float) * min(weight, 5.0) * artist_scale

    top_tracks = sorted(profile.get("track_affinity", {}).items(), key=lambda item: item[1], reverse=True)[:12]
    for track, weight in top_tracks:
        if weight <= 0:
            continue
        score += track_col.str.contains(re.escape(track), case=False, na=False).astype(float) * min(weight, 4.0) * track_scale

    top_genres = sorted(profile.get("genre_affinity", {}).items(), key=lambda item: item[1], reverse=True)[:10]
    for genre, weight in top_genres:
        if weight <= 0:
            continue
        score += genre_col.str.contains(re.escape(genre), case=False, na=False).astype(float) * min(weight, 4.0) * genre_scale

    for liked in profile.get("liked_songs", [])[:24]:
        if liked.get("track_name"):
            score += track_col.str.contains(re.escape(liked["track_name"]), case=False, na=False).astype(float) * (1.8 if directed_prompt else 2.2)
        if liked.get("artists"):
            score += artist_col.str.contains(re.escape(liked["artists"]), case=False, na=False).astype(float) * (1.6 if directed_prompt else 2.0)
        if liked.get("genre"):
            score += genre_col.str.contains(re.escape(normalize_affinity_label(liked["genre"])), case=False, na=False).astype(float) * (0.4 if directed_prompt else 1.2)

    us_genre_bonus = pd.Series(0.0, index=catalog.index, dtype=float)
    for token in US_HIT_GENRE_TOKENS:
        us_genre_bonus += genre_col.str.contains(re.escape(token), case=False, na=False).astype(float) * 0.18

    world_penalty = pd.Series(0.0, index=catalog.index, dtype=float)
    for token in DEPRIORITIZED_WORLD_GENRE_TOKENS:
        world_penalty += genre_col.str.contains(re.escape(token), case=False, na=False).astype(float) * 0.16

    score += popularity_scale * popularity_rank
    score += us_genre_bonus
    score -= world_penalty
    return score


def preference_matches(catalog: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    if catalog.empty:
        return catalog.iloc[0:0]

    scored = catalog.copy()
    scored["preference_score"] = catalog_preference_scores(scored, profile)
    matched = scored.loc[scored["preference_score"] > 0].copy()
    if matched.empty:
        return matched
    return matched.sort_values(["preference_score", "popularity"], ascending=[False, False]).head(600)


def taste_constraints(profile: dict[str, Any], catalog: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], int]:
    matched = preference_matches(catalog, profile)
    if matched.empty:
        return {}, 0

    weights = matched["preference_score"].clip(lower=0.1) + matched["popularity"].rank(method="average", pct=True) * 1.5
    constraints: dict[str, tuple[float, float]] = {}
    for feature in ("danceability", "energy", "valence", "acousticness", "instrumentalness"):
        mean = float(np.average(matched[feature].to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))
        spread = 0.17
        constraints[feature] = (max(0.0, mean - spread), min(1.0, mean + spread))

    tempo_mean = float(np.average(matched["tempo"].to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))
    constraints["tempo_bpm"] = (max(60.0, tempo_mean - 18.0), min(190.0, tempo_mean + 18.0))
    return constraints, len(matched)


def blend_constraint_ranges(
    base: dict[str, tuple[float, float]],
    taste: dict[str, tuple[float, float]],
    strength: float = 0.52,
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


def taste_blend_strength(spec: QuerySpec) -> float:
    if spec.genres:
        return 0.14
    if spec.moods or spec.constraints:
        return 0.18
    if spec.seed_artists or spec.seed_track:
        return 0.24
    return 0.34


def apply_taste_profile(spec: QuerySpec, catalog: pd.DataFrame) -> tuple[QuerySpec, str | None]:
    profile = current_taste_profile()
    taste_prior, matched_count = taste_constraints(profile, catalog)
    if not taste_prior:
        return clone_spec(spec), None

    updated = clone_spec(spec)
    blend_strength = taste_blend_strength(spec)
    updated.constraints = blend_constraint_ranges(updated.constraints, taste_prior, strength=blend_strength)
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    top_genres = top_affinity_items(profile.get("genre_affinity", {}), limit=2)
    top_artists = top_affinity_items(profile.get("artist_affinity", {}), limit=2)
    parts = []
    if top_genres:
        parts.append("genres: " + ", ".join(top_genres))
    if top_artists:
        parts.append("artists: " + ", ".join(top_artists))
    blend_note = "lightly blended" if prompt_has_clear_direction(spec) else "blended"
    note = f"I {blend_note} your listening history from **{matched_count} matched tracks** ({'; '.join(parts) or 'general listening history'})."
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


def spotify_user_token() -> str | None:
    auth = st.session_state.get("spotify_auth_session") or {}
    access_token = auth.get("access_token")
    expires_at = float(auth.get("expires_at", 0.0) or 0.0)
    if access_token and expires_at > time.time() + 30:
        return access_token

    refresh_token = auth.get("refresh_token")
    if not refresh_token:
        return None

    refreshed = spotify_refresh_user_token(refresh_token)
    if not refreshed:
        return None
    auth.update(refreshed)
    st.session_state["spotify_auth_session"] = auth
    return auth.get("access_token")


def spotify_token_exchange(data: dict[str, str]) -> dict[str, Any] | None:
    client_id, client_secret = spotify_client_credentials()
    if not client_id or not client_secret:
        return None
    credentials = f"{client_id}:{client_secret}".encode("utf-8")
    encoded = base64.b64encode(credentials).decode("utf-8")
    payload = urllib_parse.urlencode(data).encode("utf-8")
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
            return json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError):
        return None


def spotify_exchange_auth_code(code: str) -> dict[str, Any] | None:
    payload = spotify_token_exchange(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": spotify_redirect_uri(),
        }
    )
    if not payload or not payload.get("access_token"):
        return None
    expires_in = int(payload.get("expires_in", 3600))
    return {
        "access_token": payload.get("access_token"),
        "refresh_token": payload.get("refresh_token"),
        "scope": payload.get("scope", ""),
        "expires_at": time.time() + max(60, expires_in - 60),
    }


def spotify_refresh_user_token(refresh_token: str) -> dict[str, Any] | None:
    payload = spotify_token_exchange(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    )
    if not payload or not payload.get("access_token"):
        return None
    expires_in = int(payload.get("expires_in", 3600))
    refreshed = {
        "access_token": payload.get("access_token"),
        "expires_at": time.time() + max(60, expires_in - 60),
    }
    if payload.get("refresh_token"):
        refreshed["refresh_token"] = payload.get("refresh_token")
    if payload.get("scope"):
        refreshed["scope"] = payload.get("scope")
    return refreshed


def spotify_login_url() -> str | None:
    client_id, client_secret = spotify_client_credentials()
    if not client_id or not client_secret:
        return None
    oauth_state = st.session_state.get("spotify_oauth_state_session") or secrets.token_urlsafe(24)
    st.session_state["spotify_oauth_state_session"] = oauth_state
    params = {
        "response_type": "code",
        "client_id": client_id,
        "scope": SPOTIFY_OAUTH_SCOPES,
        "redirect_uri": spotify_redirect_uri(),
        "state": oauth_state,
        "show_dialog": "false",
    }
    return f"https://accounts.spotify.com/authorize?{urllib_parse.urlencode(params)}"


def handle_spotify_oauth_callback() -> None:
    query_params = st.query_params
    if "error" in query_params:
        st.session_state["last_feedback"] = f"Spotify login failed: {query_params.get('error')}."
        query_params.clear()
        return

    code = query_params.get("code")
    if not code:
        return

    returned_state = query_params.get("state", "")
    expected_state = st.session_state.get("spotify_oauth_state_session", "")
    if expected_state and returned_state != expected_state:
        st.session_state["last_feedback"] = "Spotify login failed because the returned state token did not match."
        query_params.clear()
        return

    auth = spotify_exchange_auth_code(str(code))
    query_params.clear()
    if not auth:
        st.session_state["last_feedback"] = "Spotify login failed during token exchange."
        return

    state = current_taste_profile()
    state["spotify_user_seeded"] = False
    state["spotify_synced_track_ids"] = []
    st.session_state["spotify_auth_session"] = auth
    st.session_state["spotify_oauth_state_session"] = ""
    reset_session()

    sync_note = sync_spotify_user_preferences(force=True)
    st.session_state["last_feedback"] = sync_note or "Spotify connected."


def spotify_connected() -> bool:
    return spotify_user_token() is not None


def spotify_api_request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int | None, Any | None]:
    if params:
        url = f"{url}?{urllib_parse.urlencode(params)}"
    payload = None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib_request.Request(
        url,
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp is not None else ""
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw or None
        return exc.code, parsed
    except (urllib_error.URLError, TimeoutError):
        return None, None


def spotify_api_get(url: str, token: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
    status, payload = spotify_api_request(url, token, params=params)
    if status != 200:
        return None
    return payload if isinstance(payload, dict) else None


def spotify_paginated_items(url: str, token: str, params: dict[str, str] | None = None, pages: int = 1) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    next_url = url
    next_params = dict(params or {})
    remaining = max(1, pages)
    while next_url and remaining > 0:
        payload = spotify_api_get(next_url, token, next_params)
        if not payload:
            break
        batch = payload.get("items")
        if isinstance(batch, list):
            items.extend(item for item in batch if isinstance(item, dict))
        next_url = payload.get("next")
        next_params = None
        remaining -= 1
    return items


def merge_liked_payloads(existing: list[dict[str, Any]], new_items: list[dict[str, Any]], limit: int = 48) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for item in new_items + existing:
        key = (item.get("track_id"), item.get("track_name"), item.get("artists"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:limit]


def save_track_to_spotify_library(song) -> bool:
    token = spotify_user_token()
    if not token or not getattr(song, "track_id", ""):
        return False
    track_id = str(song.track_id)
    status, _ = spotify_api_request(
        "https://api.spotify.com/v1/me/tracks",
        token,
        method="PUT",
        params={"ids": track_id},
    )
    if status in (200, 201, 204):
        state = current_taste_profile()
        saved_ids = set(state.get("spotify_saved_track_ids", []))
        saved_ids.add(track_id)
        state["spotify_saved_track_ids"] = sorted(saved_ids)
        save_ui_state(state)
        return True
    return False


def spotify_track_saved(song) -> bool | None:
    token = spotify_user_token()
    track_id = getattr(song, "track_id", "")
    if not token or not track_id:
        return None
    state = current_taste_profile()
    if track_id in set(state.get("spotify_saved_track_ids", [])):
        return True

    status, payload = spotify_api_request(
        "https://api.spotify.com/v1/me/tracks/contains",
        token,
        params={"ids": track_id},
    )
    if status == 200 and isinstance(payload, list) and payload:
        saved = bool(payload[0])
        if saved:
            saved_ids = set(state.get("spotify_saved_track_ids", []))
            saved_ids.add(track_id)
            state["spotify_saved_track_ids"] = sorted(saved_ids)
            save_ui_state(state)
        return saved
    return None


def sync_spotify_user_preferences(force: bool = False) -> str | None:
    state = current_taste_profile()
    if state.get("spotify_user_seeded") and not force:
        summary = state.get("spotify_user_summary") or {}
        if summary and summary.get("genres"):
            artists = ", ".join(summary.get("artists", [])[:3])
            genres = ", ".join(summary.get("genres", [])[:3])
            return f"Spotify history loaded: {artists or 'artists unavailable'} | {genres or 'genres unavailable'}."
        force = True

    token = spotify_user_token()
    if not token:
        return None

    short_artists = spotify_api_get("https://api.spotify.com/v1/me/top/artists", token, {"limit": "15", "time_range": "short_term"})
    medium_artists = spotify_api_get("https://api.spotify.com/v1/me/top/artists", token, {"limit": "20", "time_range": "medium_term"})
    long_artists = spotify_api_get("https://api.spotify.com/v1/me/top/artists", token, {"limit": "20", "time_range": "long_term"})
    short_tracks = spotify_api_get("https://api.spotify.com/v1/me/top/tracks", token, {"limit": "15", "time_range": "short_term"})
    medium_tracks = spotify_api_get("https://api.spotify.com/v1/me/top/tracks", token, {"limit": "20", "time_range": "medium_term"})
    long_tracks = spotify_api_get("https://api.spotify.com/v1/me/top/tracks", token, {"limit": "20", "time_range": "long_term"})
    saved_tracks = spotify_paginated_items("https://api.spotify.com/v1/me/tracks", token, {"limit": "50"}, pages=2)
    recent_tracks = spotify_paginated_items("https://api.spotify.com/v1/me/player/recently-played", token, {"limit": "50"}, pages=1)

    if not any((short_artists, medium_artists, long_artists, short_tracks, medium_tracks, long_tracks, saved_tracks, recent_tracks)):
        return None

    artist_weights: list[tuple[str, float]] = []
    genre_weights: list[tuple[str, float]] = []
    track_weights: list[tuple[str, float]] = []
    liked_payloads: list[dict[str, Any]] = []
    saved_track_ids: set[str] = set()
    track_payloads: list[tuple[dict[str, Any], float]] = []

    def ingest_artists(payload: dict[str, Any] | None, artist_boost: float, genre_boost: float) -> None:
        for artist in (payload or {}).get("items", []) or []:
            name = artist.get("name")
            if name:
                artist_weights.append((name, artist_boost))
            for genre in artist.get("genres") or []:
                genre_weights.append((genre, genre_boost))

    def ingest_track_item(track: dict[str, Any], boost: float, liked: bool) -> None:
        track_id = track.get("id", "")
        if track_id:
            saved_track_ids.add(track_id)
        name = track.get("name")
        if name:
            track_weights.append((name, boost))
        artist_names = [artist.get("name", "") for artist in track.get("artists", []) if artist.get("name")]
        for artist_name in artist_names:
            artist_weights.append((artist_name, boost * 0.9))
        liked_payloads.append(
            {
                "track_id": track_id,
                "track_name": track.get("name", ""),
                "artists": ";".join(artist_names),
                "album_name": (track.get("album") or {}).get("name", ""),
                "genre": "",
            }
        )
        track_payloads.append((track, boost))
        if liked:
            for artist_name in artist_names:
                artist_weights.append((artist_name, 1.4))

    ingest_artists(short_artists, artist_boost=3.8, genre_boost=2.8)
    ingest_artists(medium_artists, artist_boost=3.0, genre_boost=2.2)
    ingest_artists(long_artists, artist_boost=2.5, genre_boost=1.9)

    for payload, boost in ((short_tracks, 3.6), (medium_tracks, 2.8), (long_tracks, 2.3)):
        for track in (payload or {}).get("items", []) or []:
            ingest_track_item(track, boost=boost, liked=True)

    for item in saved_tracks:
        track = item.get("track") or {}
        if isinstance(track, dict):
            ingest_track_item(track, boost=2.9, liked=True)

    for item in recent_tracks:
        track = item.get("track") or {}
        if isinstance(track, dict):
            ingest_track_item(track, boost=1.4, liked=False)

    if not genre_weights:
        catalog = load_catalog()
        catalog_with_id = catalog.assign(track_id_str=catalog["track_id"].fillna("").astype(str))
        genre_col = catalog_with_id["track_genre"].fillna("").astype(str)
        name_col = catalog_with_id["track_name"].fillna("").astype(str).str.lower()
        artist_col = catalog_with_id["artists"].fillna("").astype(str).str.lower()
        for track, boost in track_payloads:
            track_id = str(track.get("id", "") or "")
            matched_rows = catalog_with_id.iloc[0:0]
            if track_id:
                matched_rows = catalog_with_id.loc[catalog_with_id["track_id_str"] == track_id]
            if matched_rows.empty:
                track_name = str(track.get("name", "") or "").strip().lower()
                artist_names = [str(artist.get("name", "")).strip().lower() for artist in track.get("artists", []) if artist.get("name")]
                if track_name:
                    matched_rows = catalog_with_id.loc[name_col == track_name]
                if not matched_rows.empty and artist_names:
                    artist_pattern = "|".join(re.escape(name) for name in artist_names[:2] if name)
                    if artist_pattern:
                        matched_rows = matched_rows.loc[
                            matched_rows["artists"].fillna("").astype(str).str.lower().str.contains(artist_pattern, regex=True, na=False)
                        ]
            if matched_rows.empty:
                continue
            top_genres = (
                matched_rows.assign(_genre=genre_col.loc[matched_rows.index])
                .groupby("_genre")["popularity"]
                .mean()
                .sort_values(ascending=False)
                .head(2)
                .index
                .tolist()
            )
            for genre in top_genres:
                if genre:
                    genre_weights.append((str(genre), boost * 0.9))
            if top_genres and track_id:
                for liked in liked_payloads:
                    if liked.get("track_id") == track_id and not liked.get("genre"):
                        liked["genre"] = str(top_genres[0])

    for artist_name, weight in artist_weights:
        bump_affinity(state.setdefault("artist_affinity", {}), [artist_name], weight)
    for genre_name, weight in genre_weights:
        bump_affinity(state.setdefault("genre_affinity", {}), [genre_name], weight)
    for track_name, weight in track_weights:
        bump_affinity(state.setdefault("track_affinity", {}), [track_name], weight)

    existing = state.setdefault("liked_songs", [])
    state["liked_songs"] = merge_liked_payloads(existing, liked_payloads, limit=48)
    state["spotify_saved_track_ids"] = sorted(set(state.get("spotify_saved_track_ids", [])) | saved_track_ids)
    state["spotify_user_seeded"] = True
    state["spotify_user_summary"] = {
        "artists": top_affinity_items(state.get("artist_affinity", {}), limit=5),
        "genres": top_affinity_items(state.get("genre_affinity", {}), limit=5),
        "tracks": top_affinity_items(state.get("track_affinity", {}), limit=5),
    }
    save_ui_state(state)

    artists = ", ".join(state["spotify_user_summary"]["artists"][:3]) or "top artists"
    genres = ", ".join(state["spotify_user_summary"]["genres"][:3]) or "top genres"
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
    spec = enrich_spec_from_prompt(prompt, get_parser().parse(prompt))
    infer_preferences_from_message(prompt, spec)
    spec, taste_note = apply_taste_profile(spec, load_catalog())
    st.session_state["seen_track_ids"] = set()
    session = build_session_from_spec(spec, playlist_length, excluded_track_ids=set())
    st.session_state["dj_session"] = session
    st.session_state["session_finished"] = session._current_song is None
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["latest_model_update"] = f"Session initialized from: {prompt}"
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    add_chat_message("user", prompt)
    response = compose_assistant_message(
        f"**You asked for:** {prompt}",
        humanize_spotify_note(spotify_note),
        humanize_taste_note(taste_note),
        summarize_spec(spec, session),
    )
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


def message_starts_new_request(message: str) -> bool:
    lowered = message.lower().strip()
    refinement_cues = ("more ", "less ", "keep ", "but ", "instead", "slower", "faster", "open up", "reset priors")
    fresh_request_cues = ("give me", "play me", "i want", "can you", "make me", "find me", "put on")
    if any(cue in lowered for cue in refinement_cues):
        return False
    return any(lowered.startswith(cue) for cue in fresh_request_cues)


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
    refinement = enrich_spec_from_prompt(message, get_parser().parse(message))
    infer_preferences_from_message(message, refinement)
    spotify_note = sync_spotify_user_preferences()
    merged_spec = merge_specs(previous_spec, refinement, message)
    merged_spec, taste_note = apply_taste_profile(merged_spec, load_catalog())
    prior_feedback = list(zip(current_session.playlist, current_session.actions))
    rebuilt_session = build_session_from_spec(
        merged_spec,
        current_session.playlist_length,
        prior_feedback=prior_feedback,
        excluded_track_ids=set(st.session_state.get("seen_track_ids", set())),
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
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    add_chat_message("user", message)

    change_summary = describe_changes(previous_spec, merged_spec)
    if not change_summary:
        change_summary = "I kept the structure mostly intact and treated your note as a softer steering signal."
    st.session_state["latest_model_update"] = change_summary
    assistant_reply = compose_assistant_message(
        f"**I'm reshaping the set:** {change_summary}",
        humanize_spotify_note(spotify_note),
        humanize_taste_note(taste_note),
        f"**I'm keeping your earlier reactions in the mix too:** {len(prior_feedback)} prior decisions are still informing this.",
        summarize_spec(merged_spec, rebuilt_session),
    )
    add_chat_message("assistant", assistant_reply)


GENRE_TRANSITIONS = {
    "indie": ["indie pop", "alternative", "folk"],
    "hip hop": ["r&b", "neo soul", "house"],
    "trance": ["melodic techno", "progressive house", "dance"],
    "house": ["disco", "dance", "soul"],
    "jazz": ["neo soul", "soul", "funk"],
    "rock": ["indie", "new wave", "alternative"],
}

INTERVENTION_CONSTRAINTS: dict[str, dict[str, tuple[float, float]]] = {
    "workout": {
        "energy": (0.74, 1.0),
        "danceability": (0.56, 1.0),
        "tempo_bpm": (118.0, 176.0),
    },
    "focus": {
        "energy": (0.24, 0.58),
        "acousticness": (0.16, 0.82),
        "instrumentalness": (0.0, 0.58),
        "tempo_bpm": (78.0, 126.0),
    },
    "confidence": {
        "energy": (0.58, 0.94),
        "valence": (0.44, 0.92),
        "danceability": (0.52, 0.92),
    },
    "late_night": {
        "energy": (0.18, 0.6),
        "valence": (0.08, 0.58),
        "acousticness": (0.1, 0.7),
        "tempo_bpm": (76.0, 122.0),
    },
    "golden_hour": {
        "energy": (0.24, 0.62),
        "valence": (0.34, 0.84),
        "acousticness": (0.12, 0.66),
    },
    "after_hours": {
        "energy": (0.22, 0.72),
        "danceability": (0.42, 0.84),
        "valence": (0.12, 0.58),
    },
    "feel_good": {
        "energy": (0.42, 0.86),
        "danceability": (0.44, 0.9),
        "valence": (0.58, 1.0),
    },
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


def choose_shift_genre(spec: QuerySpec, candidates: list[str] | None = None) -> str | None:
    current = {normalize_affinity_label(genre) for genre in spec.genres}
    ordered: list[str] = []
    for candidate in candidates or []:
        if candidate:
            ordered.append(candidate)
    fallback = next_transition_genre(spec)
    if fallback:
        ordered.append(fallback)
    for candidate in ordered:
        normalized = normalize_affinity_label(candidate)
        if not normalized or normalized in current:
            continue
        return candidate
    return fallback


def merge_route_constraints(
    base: dict[str, tuple[float | int, float | int]],
    route_name: str,
) -> dict[str, tuple[float | int, float | int]]:
    hints = INTERVENTION_CONSTRAINTS.get(route_name)
    if not hints:
        return dict(base)
    return blend_constraint_ranges(dict(base), hints, strength=0.72)


def intervention_profile_summary() -> dict[str, list[str]]:
    profile = current_taste_profile()
    spotify_summary = profile.get("spotify_user_summary") or {}
    artists = spotify_summary.get("artists") or top_affinity_items(profile.get("artist_affinity", {}), limit=5)
    genres = spotify_summary.get("genres") or top_affinity_items(profile.get("genre_affinity", {}), limit=5)
    tracks = spotify_summary.get("tracks") or top_affinity_items(profile.get("track_affinity", {}), limit=5)
    liked_genres = [
        normalize_affinity_label(liked.get("genre", ""))
        for liked in profile.get("liked_songs", [])[:12]
        if liked.get("genre")
    ]
    liked_artists = [
        normalize_affinity_label(liked.get("artists", "").split(";")[0])
        for liked in profile.get("liked_songs", [])[:12]
        if liked.get("artists")
    ]
    return {
        "artists": [item for item in artists if item],
        "genres": [item for item in genres if item],
        "tracks": [item for item in tracks if item],
        "liked_genres": [item for item in liked_genres if item],
        "liked_artists": [item for item in liked_artists if item],
    }


def build_intervention_routes(spec: QuerySpec, rounds: int) -> list[dict[str, Any]]:
    summary = intervention_profile_summary()
    top_artists = summary["artists"]
    top_genres = summary["genres"]
    top_tracks = summary["tracks"]
    liked_genres = summary["liked_genres"]
    liked_artists = summary["liked_artists"]
    routes: list[dict[str, Any]] = []

    def add_route(
        name: str,
        headline: str,
        reason: str,
        speech: str,
        builder,
    ) -> None:
        routes.append(
            {
                "name": name,
                "headline": headline,
                "reason": reason,
                "speech": speech,
                "builder": builder,
            }
        )

    pivot_genre = next_transition_genre(spec)
    if pivot_genre:
        add_route(
            "genre_pivot",
            f"I'm flipping the room toward **{pivot_genre}**.",
            "You have enough posterior signal here, so this is a controlled genre pivot rather than a reset.",
            f"Quick DJ Bayes pivot. I'm taking you into a {pivot_genre} lane for the next stretch while keeping what I've learned about your taste intact.",
            lambda base: _route_with_genre_pivot(base, pivot_genre),
        )

    if top_artists:
        artist = random.choice(top_artists[: min(4, len(top_artists))])
        artist_genre = choose_shift_genre(spec, top_genres + liked_genres)
        add_route(
            "artist_spotlight",
            f"I'm pulling from your **heavy-rotation artist lane**: {artist}" + (f", and turning it toward **{artist_genre}**." if artist_genre else "."),
            f"This route leans on your Spotify history instead of the current prompt alone.",
            f"DJ Bayes check-in. I'm going artist mode for a minute and steering this set through the lane around {artist}." if not artist_genre else f"DJ Bayes check-in. I'm going artist mode for a minute and steering this set around {artist}, but with a clear tilt into {artist_genre}.",
            lambda base, artist=artist, artist_genre=artist_genre: _route_with_artist_focus(base, artist, artist_genre),
        )

    if top_tracks:
        track = random.choice(top_tracks[: min(4, len(top_tracks))])
        rewind_genre = choose_shift_genre(spec, liked_genres + top_genres)
        add_route(
            "favorite_rewind",
            f"I'm doing a **heavy-rotation rewind** around **{track}**.",
            "This route uses what you seem to come back to most, then expands around that fingerprint.",
            f"Time for a left turn based on your repeats. I'm borrowing the feel of {track} and building the next pocket from there." if not rewind_genre else f"Time for a left turn based on your repeats. I'm borrowing the feel of {track} and flipping the genre direction toward {rewind_genre}.",
            lambda base, track=track, rewind_genre=rewind_genre: _route_with_track_memory(base, track, rewind_genre),
        )

    if top_genres:
        genre = choose_shift_genre(spec, top_genres[: min(4, len(top_genres))]) or random.choice(top_genres[: min(4, len(top_genres))])
        add_route(
            "favorite_genre",
            f"I'm reopening one of your strongest Spotify lanes: **{genre}**.",
            "This route is personalized from your top-genre history, not just the current prompt.",
            f"DJ Bayes here. I'm reopening one of your favorite lanes and giving you a {genre} run for the next few records.",
            lambda base, genre=genre: _route_with_genre_memory(base, genre),
        )

    if liked_genres:
        liked_genre = choose_shift_genre(spec, liked_genres[: min(5, len(liked_genres))]) or random.choice(liked_genres[: min(5, len(liked_genres))])
        add_route(
            "liked_lane",
            f"I'm taking a detour through a **recently liked lane**: **{liked_genre}**.",
            "This route keys off what you've positively reinforced inside the session.",
            f"I've been clocking what you keep rewarding, so I'm sliding into a {liked_genre} pocket for a few songs.",
            lambda base, liked_genre=liked_genre: _route_with_genre_memory(base, liked_genre),
        )

    workout_artist = random.choice(top_artists[: min(3, len(top_artists))]) if top_artists else None
    workout_reason = f"I'm using {workout_artist} as the anchor and raising the energy ceiling." if workout_artist else "I'm raising the energy ceiling and tightening the rhythmic profile."
    workout_speech = f"Let's change the temperature. I'm giving you a workout burst next." if not workout_artist else f"Let's change the temperature. I'm giving you a workout burst built around the energy near {workout_artist}."
    add_route(
        "workout_mode",
        "I'm switching the room into **workout mode**.",
        workout_reason,
        workout_speech,
        lambda base, artist=workout_artist: _route_with_mood_pack(base, "workout", ["workout", "hype", "confident"], artist, choose_shift_genre(base, ["trap", "house", "drum and bass", "hip hop", "rage"])),
    )

    focus_artist = random.choice(top_artists[: min(3, len(top_artists))]) if top_artists else None
    add_route(
        "focus_tunnel",
        "I'm carving out a **focus tunnel** for the next few tracks.",
        "This route lowers the chaos a bit and tries to make the stream more useful, not just exciting.",
        "Quick palette cleanse. I'm giving you a sharper focus stretch before the next lift.",
        lambda base, artist=focus_artist: _route_with_mood_pack(base, "focus", ["focus", "smooth", "driving"], artist, choose_shift_genre(base, ["neo soul", "jazz", "lofi", "indie pop", "ambient"])),
    )

    night_artist = random.choice((liked_artists or top_artists)[: min(3, len(liked_artists or top_artists))]) if (liked_artists or top_artists) else None
    add_route(
        "after_hours",
        "I'm dipping this into an **after-hours** pocket.",
        "This route is meant to feel smoother and more cinematic without losing your profile.",
        "DJ Bayes after-hours switch. I'm dimming the lights for a smoother run before I open it back up.",
        lambda base, artist=night_artist: _route_with_mood_pack(base, "after_hours", ["late night", "after hours", "smooth"], artist, choose_shift_genre(base, ["r&b", "alt r&b", "neo soul", "house"])),
    )

    add_route(
        "golden_hour",
        "I'm taking a **golden-hour detour**.",
        "This is a softer, warmer pivot to keep the stream surprising without breaking coherence.",
        "I want a little color here, so I'm taking this through a warmer golden-hour lane for a few songs.",
        lambda base: _route_with_mood_pack(base, "golden_hour", ["golden hour", "warm", "dreamy"], None, choose_shift_genre(base, ["afrobeats", "indie pop", "neo soul", "house"])),
    )

    add_route(
        "confidence_boost",
        "I'm queuing a **confidence boost** run.",
        "This route chases swagger, cleaner hits, and momentum when the set needs a lift.",
        "Time to sharpen the edges. I'm pushing this into a confidence-boost lane for the next stretch.",
        lambda base: _route_with_mood_pack(base, "confidence", ["confident", "upbeat", "party"], None, choose_shift_genre(base, ["hip hop", "pop rap", "dance pop", "house"])),
    )

    add_route(
        "feel_good_reset",
        "I'm doing a **feel-good reset**.",
        "This route leans brighter and more familiar so the stream doesn't get stuck in one emotional register.",
        "Quick reset. I'm brightening this up and giving you a more feel-good run for a few songs.",
        lambda base: _route_with_mood_pack(base, "feel_good", ["feel good", "uplifting", "summer"], None, choose_shift_genre(base, ["disco", "funk", "dance pop", "afrobeats"])),
    )

    shuffled = list(routes)
    random.shuffle(shuffled)
    return shuffled


def _route_with_genre_pivot(base: QuerySpec, genre: str) -> QuerySpec:
    updated = clone_spec(base)
    updated.genres = [genre]
    updated.moods = list(dict.fromkeys((updated.moods + ["confident", "upbeat"])[:6]))
    updated.constraints = merge_route_constraints(updated.constraints, "confidence")
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    return updated


def _route_with_artist_focus(base: QuerySpec, artist: str, genre: str | None) -> QuerySpec:
    updated = clone_spec(base)
    updated.seed_artists = [artist]
    if genre:
        updated.genres = [genre]
    updated.moods = list(dict.fromkeys((["smooth", "confident"] + updated.moods)[:6]))
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    return updated


def _route_with_track_memory(base: QuerySpec, track: str, genre: str | None) -> QuerySpec:
    updated = clone_spec(base)
    updated.seed_track = track
    if genre:
        updated.genres = [genre]
    updated.moods = list(dict.fromkeys((["nostalgic", "confident"] + updated.moods)[:6]))
    updated.constraints = merge_route_constraints(updated.constraints, "feel_good")
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    return updated


def _route_with_genre_memory(base: QuerySpec, genre: str) -> QuerySpec:
    updated = clone_spec(base)
    updated.genres = list(dict.fromkeys(([genre] + updated.genres)[:4]))
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    return updated


def _route_with_mood_pack(base: QuerySpec, route_name: str, moods: list[str], artist: str | None, genre: str | None) -> QuerySpec:
    updated = clone_spec(base)
    if genre:
        updated.genres = [genre]
    updated.moods = list(dict.fromkeys((moods + updated.moods)[:6]))
    updated.constraints = merge_route_constraints(updated.constraints, route_name)
    if artist:
        updated.seed_artists = [artist]
    updated.spotify_search_queries = updated.to_spotify_search_queries()
    return updated


def schedule_speech(text: str, key: str) -> None:
    st.session_state["speech_payload"] = {"text": text, "key": key}


def maybe_trigger_dj_interlude() -> None:
    session = st.session_state["dj_session"]
    rounds = len(session.playlist)
    if rounds == 0 or rounds % 4 != 0:
        return
    if st.session_state.get("last_transition_round") == rounds:
        return

    prior_feedback = list(zip(session.playlist, session.actions))
    catalog = load_catalog()
    route_history = st.session_state.get("recent_intervention_routes", [])
    routes = build_intervention_routes(session.spec, rounds)
    if route_history:
        filtered = [route for route in routes if route["name"] not in route_history[-2:]]
        if filtered:
            routes = filtered

    chosen_route: dict[str, Any] | None = None
    rebuilt: DJSession | None = None
    for route in routes:
        candidate_spec = route["builder"](session.spec)
        candidate_spec, _ = apply_taste_profile(candidate_spec, catalog)
        candidate_session = build_session_from_spec(
            candidate_spec,
            session.playlist_length,
            prior_feedback=prior_feedback,
            excluded_track_ids=set(st.session_state.get("seen_track_ids", set())),
        )
        if getattr(candidate_session, "initial_candidate_count", 0) <= 0 or candidate_session._current_song is None:
            continue
        chosen_route = route
        rebuilt = candidate_session
        break

    if rebuilt is not None and chosen_route is not None:
        st.session_state["dj_session"] = rebuilt
        st.session_state["session_finished"] = rebuilt._current_song is None
        st.session_state["playback_song"] = None
        st.session_state["playback_scored"] = False
        st.session_state["back_song"] = None
        st.session_state["showing_back_song"] = False
        route_history = (route_history + [chosen_route["name"]])[-6:]
        st.session_state["recent_intervention_routes"] = route_history

    st.session_state["last_transition_round"] = rounds
    if chosen_route is not None and rebuilt is not None and rebuilt._current_song is not None:
        message = compose_assistant_message(
            f"**DJ Bayes intervention:** {chosen_route['headline']}",
            f"**Why this route:** {chosen_route['reason']}",
            f"**Next up:** {rebuilt._current_song.track_name} by {rebuilt._current_song.artists}",
        )
        latest_update = f"Auto-intervention after {rounds} songs: {chosen_route['headline'].replace('**', '')}"
        speech_text = str(chosen_route["speech"])
    else:
        fallback_genre = next_transition_genre(session.spec) or "a new lane"
        message = compose_assistant_message(
            f"**DJ Bayes intervention:** I'm nudging the set toward **{fallback_genre}**.",
            "**Why this route:** I couldn't find a stronger alternate detour with enough songs, so I kept the pivot simple.",
        )
        latest_update = f"Auto-intervention after {rounds} songs toward {fallback_genre}."
        speech_text = f"DJ Bayes here. You've moved through {rounds} songs, so I'm nudging this set toward {fallback_genre} for the next stretch."
    add_chat_message("assistant", message)
    st.session_state["latest_model_update"] = latest_update
    schedule_speech(speech_text, f"transition-{rounds}")


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
        const pendingKey = "dj-bayes-pending-speech";
        if (window.sessionStorage.getItem(storeKey) !== key) {{
            window.sessionStorage.setItem(storeKey, key);
            window.sessionStorage.setItem(pendingKey, "1");
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.02;
            utterance.lang = "en-US";
            utterance.onend = () => {{
                window.sessionStorage.removeItem(pendingKey);
            }};
            utterance.onerror = () => {{
                window.sessionStorage.removeItem(pendingKey);
            }};
            window.speechSynthesis.speak(utterance);
        }} else {{
            window.sessionStorage.removeItem(pendingKey);
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
                "feedback": "Like" if action == "play" else "Don't like",
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


def related_reference_tracks(song, limit: int = 2) -> list[str]:
    profile = current_taste_profile()
    references: list[str] = []
    seen: set[str] = set()
    song_artists = {normalize_affinity_label(part) for part in song.artists.split(";") if part.strip()}
    song_genre = normalize_affinity_label(song.genre)

    for liked in profile.get("liked_songs", [])[:24]:
        track_name = liked.get("track_name", "")
        if not track_name or track_name == song.track_name or track_name in seen:
            continue
        liked_genre = normalize_affinity_label(liked.get("genre", ""))
        liked_artists = {
            normalize_affinity_label(part)
            for part in liked.get("artists", "").split(";")
            if part.strip()
        }
        if liked_genre == song_genre or (song_artists and liked_artists & song_artists):
            seen.add(track_name)
            references.append(track_name)
        if len(references) >= limit:
            return references

    spotify_summary = profile.get("spotify_user_summary") or {}
    for track_name in spotify_summary.get("tracks", [])[:5]:
        if not track_name or track_name == song.track_name or track_name in seen:
            continue
        seen.add(track_name)
        references.append(track_name)
        if len(references) >= limit:
            break
    return references


def recommendation_reason_text(song, score: float) -> str:
    references = related_reference_tracks(song, limit=2)
    if score >= 0.82:
        base = "You would likely keep this in rotation."
    elif score >= 0.64:
        base = "This should land well with your taste."
    elif score >= 0.46:
        base = "There is a decent chance this clicks for you."
    else:
        base = "This is a stretch pick, but it still overlaps with parts of your profile."
    if references:
        return f"{base} It lines up with tracks you already seem to like such as {', '.join(references)}."
    return base


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
    for message in st.session_state["chat_messages"][-8:]:
        role = "You" if message["role"] == "user" else "DJ Bayes"
        role_class = "user" if message["role"] == "user" else "assistant"
        content = render_message_html(str(message["content"]))
        st.markdown(
            f"""
            <div class="chat-bubble {role_class}">
                <div class="chat-role">{html.escape(role)}</div>
                <div class="chat-body">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
    return " | ".join(parts) if parts else "Spotify listening history has not been synced yet."


def render_latest_update() -> None:
    update = st.session_state.get("latest_model_update") or "No posterior update yet."
    preference_text = preference_summary_text()
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Latest Model Update</div>
            <div class="playlist-title"><strong>{html.escape(update)}</strong></div>
            <div class="playlist-meta" style="margin-top:0.55rem;"><strong>Spotify baseline:</strong> {html.escape(preference_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(message: str, tone: str = "info") -> None:
    st.markdown(
        f'<div class="status-banner {html.escape(tone)}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def trigger_auto_next(track_id: str) -> str:
    safe_track_id = html.escape(str(track_id))
    return f"""
        const autoNextKey = "dj-bayes-auto-next";
        function goNext() {{
            if (window.sessionStorage.getItem(autoNextKey) === "{safe_track_id}") {{
                return;
            }}
            window.sessionStorage.setItem(autoNextKey, "{safe_track_id}");
            const target = new URL(window.parent.location.href);
            target.searchParams.set("autonext", "{safe_track_id}");
            window.parent.location.href = target.toString();
        }}
    """


def maybe_handle_auto_next(session: DJSession, song, playback_song) -> bool:
    query_params = st.query_params
    auto_next_track = query_params.get("autonext")
    if not auto_next_track or str(auto_next_track) != str(getattr(song, "track_id", "")):
        return False

    query_params.clear()
    pending = st.session_state.get("pending_reaction")
    if pending and str(pending.get("track_id", "")) == str(getattr(song, "track_id", "")):
        clear_pending_reaction()
    st.session_state["back_song"] = serialize_song(song)
    if playback_song is not None:
        st.session_state["playback_song"] = None
        st.session_state["playback_scored"] = False
        st.session_state["showing_back_song"] = False
        if session._current_song is None and not session_complete(session):
            ensure_current_song(session)
    else:
        session.advance_without_feedback()
        ensure_current_song(session)
    st.session_state["last_feedback"] = f"Auto-advanced after {song.track_name} finished."
    st.session_state["session_finished"] = session._current_song is None and session_complete(session)
    return True


def render_playback_area(song, art: dict[str, str | None]) -> None:
    preview_url = art.get("preview_url")
    track_id = getattr(song, "track_id", "")
    defer_for_voice = "true" if st.session_state.get("speech_payload") else "false"
    if preview_url:
        audio_markup = f"""
        <div class="playback-shell">
            <audio id="bayes-player" src="{html.escape(preview_url)}" controls autoplay style="width:100%;"></audio>
            <script>
                const audio = document.getElementById("bayes-player");
                {trigger_auto_next(track_id)}
                const deferForVoice = {defer_for_voice};
                async function waitForVoice() {{
                    if (!deferForVoice) {{
                        return;
                    }}
                    const pendingKey = "dj-bayes-pending-speech";
                    while (window.sessionStorage.getItem(pendingKey) === "1" || window.speechSynthesis.speaking) {{
                        await new Promise((resolve) => setTimeout(resolve, 250));
                    }}
                }}
                async function startPlayback() {{
                    try {{
                        await waitForVoice();
                        const maybePromise = audio.play();
                        if (maybePromise) {{
                            await maybePromise;
                        }}
                    }} catch (err) {{}}
                }}
                startPlayback();
                audio.addEventListener("ended", goNext);
            </script>
        </div>
        """
        components.html(audio_markup, height=88)
        return

    if track_id:
        embed_track_id = html.escape(str(track_id))
        embed_markup = f"""
        <div class="playback-shell">
            <div id="spotify-embed-{embed_track_id}"></div>
            <script src="https://open.spotify.com/embed/iframe-api/v1" async></script>
            <script>
                const mountId = "spotify-embed-{embed_track_id}";
                {trigger_auto_next(track_id)}
                const deferForVoice = {defer_for_voice};
                async function waitForVoice() {{
                    if (!deferForVoice) {{
                        return;
                    }}
                    const pendingKey = "dj-bayes-pending-speech";
                    while (window.sessionStorage.getItem(pendingKey) === "1" || window.speechSynthesis.speaking) {{
                        await new Promise((resolve) => setTimeout(resolve, 250));
                    }}
                }}
                window.onSpotifyIframeApiReady = (IFrameAPI) => {{
                    const element = document.getElementById(mountId);
                    if (!element) {{
                        return;
                    }}
                    const options = {{
                        width: "100%",
                        height: 152,
                        uri: "spotify:track:{embed_track_id}"
                    }};
                    const callback = async (EmbedController) => {{
                        try {{
                            await waitForVoice();
                            EmbedController.play();
                        }} catch (err) {{}}
                        let autoAdvanced = false;
                        EmbedController.addListener("playback_update", (event) => {{
                            const data = event && event.data ? event.data : null;
                            if (!data || autoAdvanced) {{
                                return;
                            }}
                            const duration = Number(data.duration || 0);
                            const position = Number(data.position || 0);
                            if (duration > 0 && position >= duration - 1000) {{
                                autoAdvanced = true;
                                goNext();
                            }}
                        }});
                    }};
                    IFrameAPI.createController(element, options, callback);
                }};
            </script>
        </div>
        """
        components.html(embed_markup, height=182)
        return

    render_status_banner(
        "Spotify does not provide an embeddable snippet for this track.",
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

    if maybe_handle_auto_next(session, song, playback_song):
        st.rerun()

    if maybe_sync_spotify_saved_feedback(session, song):
        st.rerun()
    saved_on_spotify = spotify_track_saved(song)

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

    st.markdown(
        f"""
        <div class="glass-card" style="margin-top:0.85rem;">
            <div class="metric-label">Why This Fits</div>
            <div class="playlist-meta" style="margin-top:0.4rem;">{html.escape(recommendation_reason_text(song, posterior_score))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    related_tracks = related_reference_tracks(song, limit=2)
    if related_tracks and playback_song is None and not st.session_state.get("showing_back_song", False):
        st.markdown('<div class="section-label">Sounds Like Music You Already Like</div>', unsafe_allow_html=True)
        related_cols = st.columns(len(related_tracks))
        for idx, (column, track_name) in enumerate(zip(related_cols, related_tracks), start=1):
            with column:
                if st.button(f"More like {track_name}", key=f"related_boost_{song.track_id}_{idx}", width="stretch"):
                    queue_pending_reaction(song, f"Connected this recommendation to {track_name}.", liked=True)
                    st.session_state["last_feedback"] = f"Queued: use {track_name} as a stronger positive signal when you move to the next song."

    render_playback_area(song, art)

    if st.session_state.get("session_finished", False):
        render_status_banner(
            "Session closed. Reset from the sidebar or start a new prompt to keep training the model.",
            tone="info",
        )
    else:
        feedback_cols = st.columns(2)
        with feedback_cols[0]:
            save_disabled = playback_song is not None or st.session_state.get("showing_back_song", False) or saved_on_spotify is True
            save_label = "Already liked on Spotify" if saved_on_spotify is True else "Like this recommendation?"
            if st.button(save_label, width="stretch", type="primary", disabled=save_disabled):
                clear_pending_reaction()
                apply_positive_feedback(
                    session,
                    song,
                    "Liked this recommendation.",
                )
                st.rerun()
        with feedback_cols[1]:
            dislike_disabled = playback_song is not None or st.session_state.get("showing_back_song", False)
            if st.button("Don't like this recommendation", width="stretch", disabled=dislike_disabled):
                clear_pending_reaction()
                apply_negative_feedback(session, song, "Didn't like this recommendation.")
                st.rerun()

        if playback_song is not None:
            st.caption("Liked songs stay visible until you move to the next recommendation.")
        elif saved_on_spotify is True:
            st.caption("The Spotify player still lets you save this track in Spotify, and DJ Bayes will learn from that too.")
        elif st.session_state.get("showing_back_song", False):
            st.caption("Back is listen-only. Return to the live stream to keep training the model.")
        else:
            st.caption("Use these buttons to train DJ Bayes directly. Use the Spotify snippet itself if you also want to save the song on Spotify.")

        action_cols = st.columns(2)
        with action_cols[0]:
            back_disabled = st.session_state.get("back_song") is None or st.session_state.get("showing_back_song", False)
            if st.button("Back", width="stretch", disabled=back_disabled):
                st.session_state["playback_song"] = st.session_state.get("back_song")
                st.session_state["playback_scored"] = True
                st.session_state["showing_back_song"] = True
                previous = deserialize_song(st.session_state["back_song"])
                st.session_state["last_feedback"] = f"Back to {previous.track_name}."
                st.rerun()
        with action_cols[1]:
            if st.button("Next", width="stretch"):
                st.session_state["back_song"] = serialize_song(song)
                if playback_song is not None:
                    clear_pending_reaction()
                    st.session_state["playback_song"] = None
                    st.session_state["playback_scored"] = False
                    st.session_state["showing_back_song"] = False
                    ensure_current_song(session)
                    st.session_state["last_feedback"] = f"Queued the next recommendation after {song.track_name}."
                else:
                    if not apply_pending_reaction_if_ready(session, song):
                        session.advance_without_feedback()
                        ensure_current_song(session)
                        st.session_state["last_feedback"] = f"Queued the next recommendation after {song.track_name}."
                st.session_state["session_finished"] = session._current_song is None and session_complete(session)
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
        render_metric_card("Liked / disliked", f"{liked_count} / {skipped_count}", "Only explicit like and don't-like feedback updates the posterior.")
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
            Spotify is connected. Ask for a vibe, scene, or energy level and DJ Bayes will start the stream from your listening history.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_spotify_login_gate() -> None:
    st.markdown('<div class="section-label">Spotify Login</div>', unsafe_allow_html=True)
    if st.session_state.get("last_feedback"):
        render_status_banner(st.session_state["last_feedback"], tone="info")
    client_id, client_secret = spotify_client_credentials()
    if not client_id or not client_secret:
        render_status_banner(
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET before launching the app.",
            tone="warning",
        )
        return

    login_url = spotify_login_url()
    st.markdown(
        """
        <div class="auth-shell">
            <div class="playlist-title"><strong>Connect Spotify to start DJ Bayes</strong></div>
            <div class="playlist-meta" style="margin-top:0.6rem;">
                DJ Bayes now requires Spotify login so it can read your top artists, recent listening,
                and saved tracks before making recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if login_url:
        st.link_button("Connect Spotify", login_url, width="stretch")
    st.caption(f"Redirect URI: {spotify_redirect_uri()}")


def main() -> None:
    inject_styles()
    init_state()
    handle_spotify_oauth_callback()
    if not spotify_connected():
        render_spotify_login_gate()
        return

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

    follow_up = st.chat_input("Ask DJ Bayes for a vibe, mood, or scene")
    if follow_up:
        with st.spinner("Updating DJ Bayes..."):
            if st.session_state["dj_session"] is None:
                start_session(follow_up, st.session_state["playlist_length"])
            elif message_starts_new_request(follow_up):
                start_session(follow_up, st.session_state["playlist_length"])
            else:
                apply_refinement(follow_up)
        st.rerun()

    render_voice_interlude()


if __name__ == "__main__":
    main()
