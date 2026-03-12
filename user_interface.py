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

from bayesian_dj.session import DJSession
from bayesian_dj.discovery import (
    DEFAULT_DISCOVERY_WEIGHTS,
    discovery_score_frame,
    update_beta_bucket,
)
from bayesian_dj.prompt_intent import PromptIntent, parse_prompt_intent
from bayesian_dj.model import BayesianLogisticRegression
from bayesian_dj.song_pool import AUDIO_FEATURES, SongPool, filter_non_adult_catalog_df
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


@st.cache_resource
def get_parser() -> MusicQueryParser:
    return MusicQueryParser()


def normalize_artist_name(value: str) -> str:
    normalized = normalize_affinity_label(value)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def catalog_artist_lookup() -> tuple[list[str], dict[str, str]]:
    """Return artist lookup built from the user's Spotify taste profile."""
    profile = current_taste_profile()
    artists_by_weight = sorted(
        profile.get("artist_affinity", {}).items(), key=lambda kv: kv[1], reverse=True
    )
    mapping: dict[str, str] = {}
    ordered: list[str] = []
    for artist, _ in artists_by_weight:
        normalized = normalize_artist_name(artist)
        if len(normalized) < 3 or normalized in mapping:
            continue
        mapping[normalized] = artist
        ordered.append(normalized)
    for alias, canonical in ARTIST_NAME_ALIASES.items():
        mapping.setdefault(normalize_artist_name(alias), canonical)
        ordered.insert(0, normalize_artist_name(alias))
    return ordered, mapping


def default_ui_state() -> dict[str, Any]:
    return {
        "artist_affinity": {},
        "genre_affinity": {},
        "track_affinity": {},
        "artist_posterior": {},
        "genre_posterior": {},
        "track_posterior": {},
        "novelty_posterior": {"alpha": 6.5, "beta": 3.5},
        "popularity_posterior": {"alpha": 5.8, "beta": 4.2},
        "recent_positive_examples": [],
        "recent_negative_examples": [],
        "session_prompt_history": [],
        "liked_songs": [],
        "spotify_art_cache": {},
        "spotify_user_seeded": False,
        "spotify_user_summary": {},
        "spotify_saved_track_ids": [],
        "spotify_recent_track_ids": [],
        "spotify_top_track_ids": [],
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
    return os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8501")


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
    st.session_state["completed_song_count"] = 0
    st.session_state["completed_song_keys"] = set()
    st.session_state["recent_intervention_routes"] = []
    st.session_state["latest_model_update"] = ""
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    st.session_state["seen_track_ids"] = set()
    st.session_state["seen_track_signatures"] = set()
    st.session_state["pending_reaction"] = None
    st.session_state["queued_prompt"] = ""
    st.session_state["spotify_auth_session"] = {}
    st.session_state["spotify_oauth_state_session"] = ""
    st.session_state["welcome_greeted"] = False
    st.session_state["_fresh_init_done"] = True


def reset_session() -> None:
    st.session_state["dj_session"] = None
    st.session_state["session_finished"] = False
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["is_playing"] = False
    st.session_state["speech_payload"] = None
    st.session_state["last_transition_round"] = 0
    st.session_state["completed_song_count"] = 0
    st.session_state["completed_song_keys"] = set()
    st.session_state["recent_intervention_routes"] = []
    st.session_state["latest_model_update"] = ""
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    st.session_state["seen_track_ids"] = set()
    st.session_state["seen_track_signatures"] = set()
    st.session_state["pending_reaction"] = None
    st.session_state["queued_prompt"] = ""
    st.session_state["welcome_greeted"] = False


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
        if song is None and replenish_session_pool(session):
            song = session.recommend_next()
        if song is None and replenish_session_pool(session):
            song = session.recommend_next()
        st.session_state["session_finished"] = song is None
        if song is not None and getattr(song, "track_id", ""):
            seen = set(st.session_state.get("seen_track_ids", set()))
            seen.add(str(song.track_id))
            st.session_state["seen_track_ids"] = seen
        if song is not None:
            signatures = set(st.session_state.get("seen_track_signatures", set()))
            signatures.add(f"{song.track_name.strip().lower()}__{song.artists.strip().lower()}")
            st.session_state["seen_track_signatures"] = signatures


def song_progress_key(song) -> str:
    track_id = str(getattr(song, "track_id", "") or "").strip()
    if track_id:
        return f"id:{track_id}"
    name = str(getattr(song, "track_name", "") or "").strip().lower()
    artists = str(getattr(song, "artists", "") or "").strip().lower()
    return f"sig:{name}__{artists}"


def mark_song_completed(song) -> bool:
    key = song_progress_key(song)
    if not key:
        return False
    completed = set(st.session_state.get("completed_song_keys", set()))
    if key in completed:
        return False
    completed.add(key)
    st.session_state["completed_song_keys"] = completed
    st.session_state["completed_song_count"] = int(st.session_state.get("completed_song_count", 0)) + 1
    return True


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


def fallback_audio_features(spec: QuerySpec) -> dict[str, float]:
    defaults: dict[str, float] = {
        "danceability": 0.66,
        "energy": 0.64,
        "loudness": -8.5,
        "speechiness": 0.10,
        "acousticness": 0.18,
        "instrumentalness": 0.03,
        "liveness": 0.18,
        "valence": 0.56,
        "tempo": 118.0,
    }
    for feature, bounds in (spec.constraints or {}).items():
        midpoint = float(bounds[0] + bounds[1]) / 2.0
        if feature == "tempo_bpm":
            defaults["tempo"] = midpoint
        elif feature in defaults:
            defaults[feature] = midpoint
    return defaults


def fetch_spotify_song_pool(
    token: str,
    spec: QuerySpec,
    intent: PromptIntent | None = None,
    n_target: int = 900,
) -> pd.DataFrame:
    """Build a discovery-heavy Spotify candidate pool.

    Spotify history shapes the prior and seeds the search space, but the pool itself
    is intentionally tilted toward tracks the user has not already saved or replayed.
    """
    intent = intent or PromptIntent()
    tracks: dict[str, dict[str, Any]] = {}
    top_all_time_tracks: list[dict[str, Any]] = []
    top_track_ids: set[str] = set(current_taste_profile().get("spotify_top_track_ids", []))
    saved_track_ids: set[str] = set(current_taste_profile().get("spotify_saved_track_ids", []))
    recent_track_ids: set[str] = set(current_taste_profile().get("spotify_recent_track_ids", []))
    top_artist_names: list[str] = []

    def _track_prompt_score(track: dict[str, Any], query: str = "", genre_tag: str = "") -> float:
        score = 0.0
        track_name = str(track.get("name", "") or "").lower()
        artist_names = " ".join(
            str(artist.get("name", "") or "").lower()
            for artist in track.get("artists", [])
            if isinstance(artist, dict)
        )
        query_text = (query or "").lower()
        genre_text = (genre_tag or "").lower()
        if query_text and query_text in track_name:
            score += 0.75
        if query_text and query_text in artist_names:
            score += 0.65
        if genre_text and genre_text in query_text:
            score += 0.15
        if intent.seed_artists and any(artist.lower() in artist_names for artist in intent.seed_artists[:3]):
            score += 0.75 if intent.direct_match_weight > 0.6 else 0.55
        if intent.seed_tracks and any(track_ref.lower() in track_name for track_ref in intent.seed_tracks[:2]):
            score += 0.60
        if genre_tag and intent.genres:
            normalized = {normalize_affinity_label(item) for item in intent.genres}
            if normalize_affinity_label(genre_tag) in normalized:
                score += 0.35
        if any(term.lower() in track_name or term.lower() in artist_names for term in intent.semantic_terms[:6]):
            score += 0.16
        return min(score, 1.0)

    def _ingest_track_obj(
        track: dict[str, Any],
        *,
        genre_tag: str = "",
        source_type: str = "search",
        prompt_query: str = "",
    ) -> None:
        tid = str(track.get("id", "") or "")
        if not tid or tid in tracks:
            return
        artist_names = [a.get("name", "") for a in track.get("artists", []) if a.get("name")]
        normalized_artists = {normalize_affinity_label(name) for name in artist_names}
        excluded = {normalize_affinity_label(name) for name in intent.excluded_artists}
        if excluded and normalized_artists & excluded:
            return
        artist_ids = [str(a.get("id", "") or "") for a in track.get("artists", []) if a.get("id")]
        release_date = ((track.get("album") or {}).get("release_date") or "").strip()
        release_year = int(release_date[:4]) if len(release_date) >= 4 and release_date[:4].isdigit() else 0
        is_saved = tid in saved_track_ids
        is_recent = tid in recent_track_ids
        is_top_track = tid in top_track_ids
        novelty_score = 0.78 + 0.28 * intent.exploration_weight
        if is_saved:
            novelty_score -= 0.28
        if is_recent:
            novelty_score -= 0.18
        if is_top_track:
            novelty_score -= 0.14
        if source_type == "history_anchor":
            novelty_score -= 0.12
        elif source_type == "related_artist_top":
            novelty_score += 0.14
        elif source_type == "recommendation":
            novelty_score += 0.10
        elif source_type == "adjacent_search":
            novelty_score += 0.18
        tracks[tid] = {
            "track_id": tid,
            "track_name": track.get("name", ""),
            "artists": ";".join(artist_names),
            "artist_ids": ";".join(artist_ids),
            "album_name": (track.get("album") or {}).get("name", ""),
            "track_genre": genre_tag,
            "popularity": track.get("popularity", 50),
            "artist_popularity": 0,
            "release_year": release_year,
            "source_type": source_type,
            "prompt_score": _track_prompt_score(track, query=prompt_query, genre_tag=genre_tag),
            "is_saved": is_saved,
            "is_recent": is_recent,
            "is_top_track": is_top_track,
            "novelty_score": float(np.clip(novelty_score, 0.0, 1.0)),
        }

    def _search_tracks(query: str, genre_tag: str = "") -> None:
        if not query.strip():
            return
        result = spotify_api_get(
            "https://api.spotify.com/v1/search",
            token,
            {
                "q": query,
                "type": "track",
                "limit": "50",
                "market": "US",
            },
        )
        for track in (result or {}).get("tracks", {}).get("items", []) or []:
            _ingest_track_obj(track, genre_tag=genre_tag, source_type="search", prompt_query=query)

    def _resolve_artist_ids(names: list[str]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for name in names[:8]:
            result = spotify_api_get(
                "https://api.spotify.com/v1/search",
                token,
                {"q": name, "type": "artist", "limit": "1", "market": "US"},
            )
            items = (result or {}).get("artists", {}).get("items", [])
            if items:
                artist_id = str(items[0].get("id", "") or "")
                if artist_id and artist_id not in seen:
                    seen.add(artist_id)
                    ids.append(artist_id)
        return ids

    def _fetch_artist_top_tracks(artist_id: str, source_type: str, genre_tag: str = "") -> None:
        result = spotify_api_get(
            f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks",
            token,
            {"market": "US"},
        )
        for track in (result or {}).get("tracks", []) or []:
            _ingest_track_obj(track, genre_tag=genre_tag, source_type=source_type, prompt_query=genre_tag)

    def _fetch_related_artist_tracks(artist_id: str, genre_tag: str = "") -> None:
        result = spotify_api_get(
            f"https://api.spotify.com/v1/artists/{artist_id}/related-artists",
            token,
        )
        for artist in ((result or {}).get("artists") or [])[:8]:
            related_id = str(artist.get("id", "") or "")
            if related_id:
                _fetch_artist_top_tracks(related_id, "related_artist_top", genre_tag)

    long_tracks = spotify_api_get(
        "https://api.spotify.com/v1/me/top/tracks",
        token,
        {"limit": "30", "time_range": "long_term"},
    )
    medium_tracks = spotify_api_get(
        "https://api.spotify.com/v1/me/top/tracks",
        token,
        {"limit": "20", "time_range": "medium_term"},
    )
    short_tracks = spotify_api_get(
        "https://api.spotify.com/v1/me/top/tracks",
        token,
        {"limit": "15", "time_range": "short_term"},
    )
    top_artists_payload = spotify_api_get(
        "https://api.spotify.com/v1/me/top/artists",
        token,
        {"limit": "12", "time_range": "long_term"},
    )
    top_all_time_ids = [
        str(track.get("id", "") or "")
        for track in (long_tracks or {}).get("items", []) or []
        if track.get("id")
    ]
    if top_artists_payload:
        top_artist_names = [
            str(item.get("name", "") or "")
            for item in top_artists_payload.get("items", []) or []
            if item.get("name")
        ]

    history_seed_ids: list[str] = []
    for payload in (long_tracks, medium_tracks, short_tracks):
        for track in (payload or {}).get("items", []) or []:
            tid = str(track.get("id", "") or "")
            if tid:
                history_seed_ids.append(tid)
                top_track_ids.add(tid)
            if payload is long_tracks:
                top_all_time_tracks.append(track)

    for item in spotify_paginated_items("https://api.spotify.com/v1/me/tracks", token, {"limit": "50"}, pages=2):
        track = item.get("track") or {}
        if isinstance(track, dict):
            tid = str(track.get("id", "") or "")
            if tid:
                saved_track_ids.add(tid)

    for item in spotify_paginated_items("https://api.spotify.com/v1/me/player/recently-played", token, {"limit": "50"}, pages=2):
        track = item.get("track") or {}
        if isinstance(track, dict):
            tid = str(track.get("id", "") or "")
            if tid:
                recent_track_ids.add(tid)

    # Map Spotify genre seeds — Spotify uses specific slug values
    SPOTIFY_GENRE_MAP: dict[str, str] = {
        "hip hop": "hip-hop", "hip-hop": "hip-hop", "trap": "trap", "rap": "hip-hop",
        "r&b": "r-n-b", "rnb": "r-n-b", "soul": "soul", "neo soul": "soul",
        "pop": "pop", "dance pop": "dance-pop", "indie pop": "indie-pop",
        "alternative": "alternative", "alt": "alternative", "indie": "indie",
        "rock": "rock", "punk": "punk", "metal": "metal", "hard rock": "hard-rock",
        "edm": "edm", "electronic": "electronic", "house": "house", "techno": "techno",
        "drum and bass": "drum-and-bass", "dubstep": "dubstep", "ambient": "ambient",
        "jazz": "jazz", "blues": "blues", "country": "country", "folk": "folk",
        "latin": "latin", "reggae": "reggaeton", "reggaeton": "reggaeton",
        "classical": "classical", "piano": "piano", "acoustic": "acoustic",
        "afrobeats": "afrobeat", "k-pop": "k-pop", "kpop": "k-pop",
        "workout": "work-out", "party": "party", "chill": "chill",
    }
    seed_genres = []
    for g in (spec.genres or []):
        slug = SPOTIFY_GENRE_MAP.get(g.lower().replace("-", " "), g.lower().replace(" ", "-"))
        if slug and slug not in seed_genres:
            seed_genres.append(slug)
    seed_genres = seed_genres[:5]

    # MOOD → audio feature targets for better matching
    MOOD_TARGETS: dict[str, dict[str, float]] = {
        "hype": {"target_energy": 0.92, "target_danceability": 0.85, "target_valence": 0.75, "target_tempo": 140.0},
        "energetic": {"target_energy": 0.88, "target_danceability": 0.80},
        "aggressive": {"target_energy": 0.92, "target_valence": 0.35},
        "chill": {"target_energy": 0.35, "target_danceability": 0.55, "target_valence": 0.55},
        "melancholy": {"target_energy": 0.30, "target_valence": 0.20, "target_acousticness": 0.60},
        "sad": {"target_energy": 0.28, "target_valence": 0.15, "target_acousticness": 0.55},
        "happy": {"target_energy": 0.72, "target_valence": 0.85, "target_danceability": 0.75},
        "focus": {"target_energy": 0.45, "target_speechiness": 0.04, "target_instrumentalness": 0.40},
        "romantic": {"target_energy": 0.45, "target_valence": 0.60, "target_acousticness": 0.45},
        "party": {"target_energy": 0.88, "target_danceability": 0.88, "target_valence": 0.80},
        "workout": {"target_energy": 0.93, "target_danceability": 0.78, "target_tempo": 145.0},
        "dark": {"target_energy": 0.65, "target_valence": 0.20, "target_acousticness": 0.20},
        "smooth": {"target_energy": 0.45, "target_danceability": 0.65, "target_valence": 0.65},
        "driving": {"target_energy": 0.75, "target_danceability": 0.70, "target_tempo": 125.0},
        "confident": {"target_energy": 0.80, "target_valence": 0.65},
        "afternoon": {"target_energy": 0.60, "target_valence": 0.65, "target_danceability": 0.68},
    }
    mood_params: dict[str, str] = {}
    for mood in (spec.moods or []):
        targets = MOOD_TARGETS.get(mood.lower(), {})
        for k, v in targets.items():
            if k not in mood_params:
                mood_params[k] = str(v)

    def _constraint_params() -> dict[str, str]:
        params: dict[str, str] = {}
        for feat, (lo, hi) in spec.constraints.items():
            mid = (lo + hi) / 2.0
            api_feat = feat if feat != "tempo_bpm" else "tempo"
            params[f"target_{api_feat}"] = f"{mid:.4f}"
        return params

    constraint_params = _constraint_params()
    # mood_params override explicit constraints for better mood fidelity
    combined_targets = {**constraint_params, **mood_params}

    requested_artist_names = list(dict.fromkeys(spec.seed_artists or []))
    seed_artist_ids = _resolve_artist_ids(requested_artist_names[:5] if requested_artist_names else top_artist_names[:3])
    history_artist_ids = _resolve_artist_ids(top_artist_names[:3]) if requested_artist_names else []

    def _fetch_batch(genres: list[str], artist_ids: list[str], genre_tag: str) -> None:
        total = len(genres) + len(artist_ids)
        if total == 0:
            return
        params: dict[str, str] = {"limit": "100", **combined_targets}
        if genres:
            params["seed_genres"] = ",".join(genres[:5])
        if artist_ids:
            remaining = 5 - len(genres)
            if remaining > 0:
                params["seed_artists"] = ",".join(artist_ids[:remaining])
        result = spotify_api_get("https://api.spotify.com/v1/recommendations", token, params)
        for track in (result or {}).get("tracks", []):
            _ingest_track_obj(track, genre_tag=genre_tag, source_type="recommendation", prompt_query=genre_tag)

    search_anchor_genres = list(dict.fromkeys((intent.genres or []) + list(spec.genres or [])))[:6]

    # Primary seeds: genre + artist combined
    if seed_genres or seed_artist_ids:
        for i in range(0, max(1, len(seed_genres)), 3):
            chunk = seed_genres[i: i + 3]
            _fetch_batch(chunk, seed_artist_ids if i == 0 else [], chunk[0] if chunk else "")
            if len(tracks) >= n_target:
                break

    prompt_artist_ids = _resolve_artist_ids(intent.seed_artists or spec.seed_artists or [])
    for artist_id in prompt_artist_ids[:3]:
        genre_tag = (intent.genres or spec.genres or [""])[0]
        _fetch_artist_top_tracks(artist_id, "seed_artist_top", genre_tag)
        _fetch_related_artist_tracks(artist_id, genre_tag)
        if len(tracks) >= n_target:
            break

    for artist_id in history_artist_ids[:4]:
        _fetch_related_artist_tracks(artist_id, (intent.genres or spec.genres or [""])[0])
        if len(tracks) >= n_target:
            break

    # Taste-profile genres as fallback filler
    if len(tracks) < n_target // 2:
        profile = current_taste_profile()
        top_genres = top_affinity_items(profile.get("genre_affinity", {}), limit=5)
        fb_genres = [SPOTIFY_GENRE_MAP.get(g.lower(), g.lower().replace(" ", "-")) for g in top_genres]
        fb_genres = [g for g in fb_genres if g]
        if fb_genres:
            _fetch_batch(fb_genres[:5], [], fb_genres[0])

    # Seed recommendations from liked songs when the pool is still thin
    if len(tracks) < n_target // 3 and history_seed_ids:
        seed_ids = list(dict.fromkeys(history_seed_ids))[:5]
        params: dict[str, str] = {"limit": "100", "seed_tracks": ",".join(seed_ids), **combined_targets}
        result = spotify_api_get("https://api.spotify.com/v1/recommendations", token, params)
        for track in (result or {}).get("tracks", []):
            _ingest_track_obj(track, source_type="recommendation", prompt_query=" ".join(spec.genres[:2] + spec.moods[:2]))

    # Search fallback: when recommendations are sparse, widen the pool with
    # direct US-market track searches based on the parsed request.
    if len(tracks) < max(160, n_target // 3):
        search_queries: list[tuple[str, str, str]] = []
        for query in spec.spotify_search_queries[:16]:
            genre_tag = (intent.genres or spec.genres or [""])[0]
            source_type = "adjacent_search" if intent.exploration_weight >= 0.7 else "search"
            search_queries.append((query, genre_tag, source_type))
        for term in intent.semantic_terms[:10]:
            genre_tag = (intent.genres or spec.genres or [""])[0]
            search_queries.append((term, genre_tag, "adjacent_search"))
        for production_tag in intent.production_tags[:5]:
            genre_tag = (intent.genres or spec.genres or [""])[0]
            search_queries.append((f"{production_tag} {genre_tag}".strip(), genre_tag, "adjacent_search"))
        for activity in intent.activity_context[:4]:
            genre_tag = (intent.genres or spec.genres or [""])[0]
            search_queries.append((f"{activity} {genre_tag}".strip(), genre_tag, "adjacent_search"))
        for artist in (intent.seed_artists or spec.seed_artists or [])[:6]:
            artist_query = artist
            if intent.moods or spec.moods:
                artist_query = f"{artist} {' '.join((intent.moods or spec.moods)[:2])}"
            search_queries.append((artist_query, (intent.genres or spec.genres or [""])[0], "search"))
        for genre in search_anchor_genres[:6]:
            search_queries.append((f"{genre} popular", genre, "search"))
            if intent.moods or spec.moods:
                search_queries.append((f"{genre} {' '.join((intent.moods or spec.moods)[:2])}", genre, "adjacent_search"))
            if intent.production_tags:
                search_queries.append((f"{genre} {' '.join(intent.production_tags[:2])}", genre, "adjacent_search"))
        for artist in (intent.seed_artists or spec.seed_artists or [])[:4]:
            search_queries.append((f"{artist} related artists", "", "adjacent_search"))
            search_queries.append((f"{artist} deep cuts", "", "adjacent_search"))
        if intent.activity_context and intent.moods:
            search_queries.append((f"{intent.activity_context[0]} {' '.join(intent.moods[:2])}", "", "adjacent_search"))

        seen_queries: set[str] = set()
        for query, genre_tag, source_type in search_queries:
            key = query.strip().lower()
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
            result = spotify_api_get(
                "https://api.spotify.com/v1/search",
                token,
                {
                    "q": query,
                    "type": "track",
                    "limit": "50",
                    "market": "US",
                },
            )
            for track in (result or {}).get("tracks", {}).get("items", []) or []:
                _ingest_track_obj(track, genre_tag=genre_tag, source_type=source_type, prompt_query=query)
            if len(tracks) >= n_target:
                break

    if len(tracks) < max(40, n_target // 8):
        for track in top_all_time_tracks[:8]:
            _ingest_track_obj(track, source_type="history_anchor", prompt_query="history anchor")
            if len(tracks) >= max(40, n_target // 8):
                break

    if not tracks:
        return pd.DataFrame()

    unique_artist_ids = sorted(
        {
            artist_id
            for row in tracks.values()
            for artist_id in str(row.get("artist_ids", "") or "").split(";")
            if artist_id
        }
    )
    artist_meta: dict[str, dict[str, Any]] = {}
    for i in range(0, len(unique_artist_ids), 50):
        chunk = unique_artist_ids[i:i + 50]
        payload = spotify_api_get("https://api.spotify.com/v1/artists", token, {"ids": ",".join(chunk)})
        for artist in (payload or {}).get("artists", []) or []:
            if isinstance(artist, dict) and artist.get("id"):
                artist_meta[str(artist["id"])] = artist

    # Fetch audio features in batches of 100
    track_ids = list(tracks.keys())
    audio_features: dict[str, dict[str, Any]] = {}
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i: i + 100]
        result = spotify_api_get(
            "https://api.spotify.com/v1/audio-features", token, {"ids": ",".join(batch)}
        )
        for feat in (result or {}).get("audio_features", []):
            if feat and feat.get("id"):
                audio_features[feat["id"]] = feat

    rows: list[dict[str, Any]] = []
    feature_defaults = fallback_audio_features(spec)
    for tid, track_data in tracks.items():
        feat = audio_features.get(tid)
        row: dict[str, Any] = {**track_data}
        artist_ids = [artist_id for artist_id in str(track_data.get("artist_ids", "")).split(";") if artist_id]
        artist_popularity = 0.0
        derived_genres: list[str] = []
        for artist_id in artist_ids[:2]:
            meta = artist_meta.get(artist_id, {})
            artist_popularity = max(artist_popularity, float(meta.get("popularity", 0) or 0))
            derived_genres.extend(str(genre) for genre in (meta.get("genres") or [])[:2])
        row["artist_popularity"] = artist_popularity
        if not row.get("track_genre") and derived_genres:
            row["track_genre"] = derived_genres[0]
        for f in AUDIO_FEATURES:
            if feat and feat.get(f) is not None:
                row[f] = float(feat.get(f, 0.0))
            else:
                row[f] = float(feature_defaults[f])
        row["_seed_all_time_rank"] = top_all_time_ids.index(tid) if tid in top_all_time_ids else -1
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_session_from_spec(
    spec: QuerySpec,
    playlist_length: int,
    prompt_intent: PromptIntent | None = None,
    prior_feedback: list[tuple[object, str]] | None = None,
    excluded_track_ids: set[str] | None = None,
    excluded_track_signatures: set[str] | None = None,
) -> tuple["DJSession", str | None]:
    """Build a DJSession backed by a Spotify-fetched song pool.

    Returns ``(session, taste_note)`` where *taste_note* is a human-readable
    string about taste-profile blending, or ``None``.
    """
    token = spotify_user_token()
    profile = current_taste_profile()

    songs_df = fetch_spotify_song_pool(token, spec, prompt_intent) if token else pd.DataFrame()

    # Apply taste-profile blending against fetched songs
    blended_spec = clone_spec(spec)
    taste_note: str | None = None
    if not songs_df.empty:
        blended_spec, taste_note = apply_taste_profile(blended_spec, songs_df)

    if songs_df.empty:
        pool = SongPool.from_songs(pd.DataFrame())
    else:
        pool = SongPool.from_songs(songs_df)
        pool.filter_by_genres(blended_spec.genres)
        if pool.n_available == 0 and len(songs_df) > 0:
            # Spotify genre labels can be sparse or imperfect. If the strict
            # genre pass wipes everything out, fall back to the fetched pool
            # instead of failing the session.
            pool = SongPool.from_songs(songs_df)
        pool.mark_used_track_ids(set(excluded_track_ids or set()))
        pool.mark_used_track_signatures(set(excluded_track_signatures or set()))
        pool.set_external_bias(
            catalog_preference_scores(songs_df, profile, blended_spec, prompt_intent).to_numpy(dtype=float)
        )

    session = DJSession(
        csv_path=None,
        pool=pool,
        playlist_length=playlist_length,
        parser=get_parser(),
    )
    session.spec = blended_spec
    session.prompt_intent = prompt_intent or PromptIntent()
    session.model = BayesianLogisticRegression.from_constraints(dict(session.spec.constraints))

    # Warm-start the prior from the user's long-term top tracks first.
    # If Spotify didn't give us enough of those, fall back to the strongest
    # taste-profile matches in the current pool.
    seeded_updates = 0
    if not songs_df.empty and "_seed_all_time_rank" in pool._df.columns:
        top_all_time = (
            pool._df.loc[pool._df["_seed_all_time_rank"] >= 0]
            .sort_values("_seed_all_time_rank")
            .head(30)
        )
        for pool_idx in top_all_time.index.tolist():
            session.model.update(spec_feature_vector(pool.get_song_info(int(pool_idx))), 1)
            seeded_updates += 1

    if seeded_updates < 20 and pool.n_available > 0:
        feat_matrix = pool.get_feature_matrix()
        bias_scores = pool.get_external_bias_scores()
        if len(bias_scores) > 0 and bias_scores.max() > 0:
            top_local = np.argsort(bias_scores)[::-1][:min(30 - seeded_updates, len(bias_scores))]
            for local_i in top_local:
                session.model.update(feat_matrix[local_i], 1)

    session.model.snapshot()
    session.initial_candidate_count = session.pool.n_available

    for song, action in prior_feedback or []:
        x = spec_feature_vector(song)
        y = 1 if action == "play" else 0
        session.model.update(x, y)
        session.model.snapshot(x=x, y=y)
        session.playlist.append(song)
        session.actions.append(action)
        session.pool.mark_song_used(
            pool_idx=getattr(song, "pool_idx", None),
            track_id=str(getattr(song, "track_id", "") or ""),
            track_name=str(getattr(song, "track_name", "") or ""),
            artists=str(getattr(song, "artists", "") or ""),
        )

    if session.pool.n_available > 0:
        if not prior_feedback and session.spec.seed_artists:
            session.recommend_next(
                preferred_artists=session.spec.seed_artists,
                require_artist_match=True,
            )
        else:
            session.recommend_next()
    return session, taste_note


def _combined_pool_df(session: DJSession, extra_df: pd.DataFrame) -> pd.DataFrame:
    existing = getattr(session.pool, "_df", pd.DataFrame()).copy()
    frames = [frame for frame in (existing, extra_df) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "track_id" in merged.columns:
        merged["track_id"] = merged["track_id"].fillna("").astype(str)
        non_empty = merged["track_id"] != ""
        with_ids = merged.loc[non_empty].drop_duplicates(subset=["track_id"], keep="first")
        without_ids = merged.loc[~non_empty].copy()
        if not without_ids.empty:
            without_ids["_sig"] = (
                without_ids["track_name"].fillna("").astype(str).str.lower().str.strip()
                + "__"
                + without_ids["artists"].fillna("").astype(str).str.lower().str.strip()
            )
            without_ids = without_ids.drop_duplicates(subset=["_sig"], keep="first").drop(columns=["_sig"])
        merged = pd.concat([with_ids, without_ids], ignore_index=True)
    return merged.reset_index(drop=True)


def replenish_session_pool(session: DJSession) -> bool:
    if session is None or getattr(session, "_current_song", None) is not None:
        return False
    token = spotify_user_token()
    if not token or session.spec is None:
        return False

    current_df = getattr(session.pool, "_df", pd.DataFrame())
    if session.pool.n_available > max(18, min(80, len(current_df) // 5 if len(current_df) else 0)):
        return False

    intent = getattr(session, "prompt_intent", PromptIntent())
    expansion_specs: list[tuple[QuerySpec, PromptIntent, int]] = []
    expansion_specs.append((clone_spec(session.spec), intent, 700))

    wider_intent = PromptIntent(**vars(intent))
    wider_intent.exploration_weight = min(0.96, float(intent.exploration_weight) + 0.12)
    wider_intent.novelty_target = min(0.96, float(intent.novelty_target) + 0.12)
    wider_intent.direct_match_weight = max(0.22, float(intent.direct_match_weight) - 0.08)
    expansion_specs.append((clone_spec(session.spec), wider_intent, 900))

    if session.spec.genres:
        relaxed_genres = clone_spec(session.spec)
        relaxed_genres.genres = list(session.spec.genres[:1])
        relaxed_genres.spotify_search_queries = relaxed_genres.to_spotify_search_queries()
        expansion_specs.append((relaxed_genres, wider_intent, 1100))

    if session.spec.constraints:
        relaxed_constraints = clone_spec(session.spec)
        relaxed_constraints.constraints = {}
        relaxed_constraints.spotify_search_queries = relaxed_constraints.to_spotify_search_queries()
        expansion_specs.append((relaxed_constraints, wider_intent, 1200))

    if session.spec.genres or session.spec.constraints:
        broad_spec = clone_spec(session.spec)
        broad_spec.genres = []
        broad_spec.constraints = {}
        broad_spec.spotify_search_queries = broad_spec.to_spotify_search_queries()
        expansion_specs.append((broad_spec, wider_intent, 1400))

    expanded = False
    base_count = len(current_df)
    for candidate_spec, candidate_intent, target_size in expansion_specs:
        extra_df = fetch_spotify_song_pool(token, candidate_spec, candidate_intent, n_target=target_size)
        merged_df = _combined_pool_df(session, extra_df)
        if merged_df.empty or len(merged_df) <= base_count:
            continue
        new_pool = SongPool.from_songs(merged_df)
        new_pool.mark_used_track_ids(set(st.session_state.get("seen_track_ids", set())))
        new_pool.mark_used_track_signatures(set(st.session_state.get("seen_track_signatures", set())))
        new_pool.set_external_bias(
            catalog_preference_scores(
                merged_df,
                current_taste_profile(),
                candidate_spec,
                candidate_intent,
            ).to_numpy(dtype=float)
        )
        session.pool = new_pool
        session.spec = candidate_spec
        session.prompt_intent = candidate_intent
        expanded = True
        break
    return expanded


def add_chat_message(role: str, content: str) -> None:
    st.session_state["chat_messages"].append({"role": role, "content": content})


def compose_assistant_message(*sections: str | None) -> str:
    cleaned = [section.strip() for section in sections if section and section.strip()]
    return "\n\n".join(cleaned)


def plain_speech_text(markdown_text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", markdown_text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    song = session._current_song
    if song is None:
        return "**Now spinning:** I’m lining up the set and opening a lane that fits your taste."

    intent = getattr(session, "prompt_intent", PromptIntent())
    tone_bits: list[str] = []
    if intent.seed_artists:
        tone_bits.append(f"around {', '.join(intent.seed_artists[:2])}")
    if intent.activity_context:
        tone_bits.append(", ".join(intent.activity_context[:1]))
    if intent.moods:
        tone_bits.append(", ".join(intent.moods[:2]))
    elif intent.genres:
        tone_bits.append(", ".join(intent.genres[:2]))

    familiarity = "fresh cuts" if not song.is_saved and not song.is_top_track else "a familiar anchor"
    lane = ", ".join(tone_bits) if tone_bits else "your lane"
    first_line = f"**Now spinning:** {song.track_name} by {song.artists}."
    second_line = f"**What I’m queuing:** {familiarity} sitting in {lane}, with enough overlap to feel right without just replaying your library."
    if intent.exploration_weight >= 0.72:
        second_line += " I’m leaning harder into discovery on this run."
    if session.last_recommendation_score is not None:
        third_line = f"**Read on the room:** {match_label(session.last_recommendation_score)}."
    else:
        third_line = None
    closing = "**Steer me live:** say things like darker, bigger hooks, more left-field, newer cuts, or closer to a different artist."
    return "\n\n".join(line for line in (first_line, second_line, third_line, closing) if line)


def parse_preference_text(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r",|\n", raw) if item.strip()]


def current_taste_profile() -> dict[str, Any]:
    return st.session_state["ui_state"]


def build_prompt_context() -> dict[str, Any]:
    session = st.session_state.get("dj_session")
    context: dict[str, Any] = {
        "recent_prompts": list(current_taste_profile().get("session_prompt_history", []))[:5],
    }
    if session is not None:
        context["last_moods"] = list(getattr(session.spec, "moods", []) or [])
        current_song = getattr(session, "_current_song", None)
        if current_song is not None:
            context["current_song_name"] = current_song.track_name
            context["current_song_artists"] = current_song.artists
            context["current_song_genre"] = current_song.genre
    return context


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


def track_signature_from_payload(track_name: str, artists: str) -> str:
    return f"{normalize_affinity_label(track_name)}__{normalize_affinity_label(artists)}"


def record_recent_example(song, liked: bool, weight: float) -> None:
    state = current_taste_profile()
    bucket_name = "recent_positive_examples" if liked else "recent_negative_examples"
    bucket = list(state.get(bucket_name, []))
    payload = {
        "track_id": str(getattr(song, "track_id", "") or ""),
        "track_name": song.track_name,
        "artists": song.artists,
        "weight": float(weight),
    }
    signature = track_signature_from_payload(song.track_name, song.artists)
    deduped = [item for item in bucket if track_signature_from_payload(item.get("track_name", ""), item.get("artists", "")) != signature]
    deduped.insert(0, payload)
    state[bucket_name] = deduped[:20]


def update_bayesian_feedback_state(song, liked: bool, *, strength: float) -> None:
    state = current_taste_profile()
    artists = [artist.strip() for artist in song.artists.split(";") if artist.strip()]
    genres = [song.genre] if getattr(song, "genre", "") else []
    update_beta_bucket(state.setdefault("artist_posterior", {}), artists, liked=liked, amount=1.15 * strength)
    update_beta_bucket(state.setdefault("genre_posterior", {}), genres, liked=liked, amount=0.9 * strength)
    update_beta_bucket(state.setdefault("track_posterior", {}), [song.track_name], liked=liked, amount=0.75 * strength)

    novelty_like = not bool(getattr(song, "is_saved", False) or getattr(song, "is_top_track", False))
    novelty_bucket = {"novelty": state.get("novelty_posterior", {"alpha": 6.5, "beta": 3.5})}
    update_beta_bucket(
        novelty_bucket,
        ["novelty"],
        liked=liked if novelty_like else not liked,
        amount=0.55 * strength,
        decay=0.998,
    )
    state["novelty_posterior"] = novelty_bucket["novelty"]

    popularity_bucket = {"mainstream": state.get("popularity_posterior", {"alpha": 5.8, "beta": 4.2})}
    mainstream_like = int(getattr(song, "popularity", 50) or 50) >= 60
    update_beta_bucket(
        popularity_bucket,
        ["mainstream"],
        liked=liked if mainstream_like else not liked,
        amount=0.40 * strength,
        decay=0.998,
    )
    state["popularity_posterior"] = popularity_bucket["mainstream"]

    record_recent_example(song, liked=liked, weight=strength)
    save_ui_state(state)


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
        combined_artists: list[str] = []
        seen_artists: set[str] = set()
        for artist in artist_mentions + enriched.seed_artists:
            key = normalize_artist_name(artist)
            if not key or key in seen_artists:
                continue
            seen_artists.add(key)
            combined_artists.append(artist)
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
    amount = 1.2 if played else -0.35
    artists = [artist.strip() for artist in song.artists.split(";") if artist.strip()]
    bump_affinity(state.setdefault("artist_affinity", {}), artists, amount)
    bump_affinity(state.setdefault("genre_affinity", {}), [song.genre], amount)
    bump_affinity(state.setdefault("track_affinity", {}), [song.track_name], amount * 0.8)
    update_bayesian_feedback_state(song, liked=played, strength=1.5 if played else 1.1)
    save_ui_state(state)


def current_feedback_for_song(session: DJSession, song) -> str | None:
    if session is None or song is None:
        return None
    current_song = getattr(session, "_current_song", None)
    if current_song is None:
        return None
    current_track_id = str(getattr(current_song, "track_id", "") or "")
    song_track_id = str(getattr(song, "track_id", "") or "")
    if current_track_id and song_track_id and current_track_id == song_track_id:
        return session.current_feedback_action()
    current_sig = track_signature_from_payload(current_song.track_name, current_song.artists)
    song_sig = track_signature_from_payload(song.track_name, song.artists)
    if current_sig == song_sig:
        return session.current_feedback_action()
    return None


def session_feedback_history(session: DJSession) -> list[tuple[object, str]]:
    history = list(zip(session.playlist, session.actions))
    current_song = getattr(session, "_current_song", None)
    current_action = session.current_feedback_action() if session is not None else None
    if current_song is not None and current_action in {"play", "skip"}:
        history.append((current_song, current_action))
    return history


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
    deltas = session.apply_feedback_to_current(True)
    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
    infer_preferences_from_song(song, played=True)
    add_song_to_liked(song)
    refresh_session_external_bias(session)
    st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
    st.session_state["latest_model_update"] = f"{source.rstrip('.')} Future recommendations now lean more toward {song.track_name} by {song.artists}, with the biggest posterior move on {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."


def apply_negative_feedback(session: DJSession, song, source: str) -> None:
    deltas = session.apply_feedback_to_current(False)
    top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
    infer_preferences_from_song(song, played=False)
    refresh_session_external_bias(session)
    st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
    st.session_state["latest_model_update"] = f"{source.rstrip('.')} Future recommendations now move away from this lane, with the biggest posterior move on {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."


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
        refresh_session_external_bias(session)
        mark_song_completed(song)
        if not maybe_trigger_dj_interlude():
            ensure_current_song(session)
        top_shift = max(deltas.items(), key=lambda item: abs(item[1]))
        st.session_state["last_feedback"] = f"{source} Largest posterior move: {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} {top_shift[1]:+.3f}."
        st.session_state["latest_model_update"] = f"{source.rstrip('.')} The posterior moved most on {AUDIO_FEATURE_LABELS.get(top_shift[0], top_shift[0].title())} ({top_shift[1]:+.3f})."
        return True

    deltas = session.record_feedback(False)
    infer_preferences_from_song(song, played=False)
    refresh_session_external_bias(session)
    mark_song_completed(song)
    if not maybe_trigger_dj_interlude():
        ensure_current_song(session)
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
        "artist_ids": getattr(song, "artist_ids", ""),
        "album_name": song.album_name,
        "genre": song.genre,
        "popularity": song.popularity,
        "artist_popularity": getattr(song, "artist_popularity", 0.0),
        "release_year": getattr(song, "release_year", 0),
        "source_type": getattr(song, "source_type", ""),
        "prompt_score": getattr(song, "prompt_score", 0.0),
        "novelty_score": getattr(song, "novelty_score", 0.5),
        "is_saved": bool(getattr(song, "is_saved", False)),
        "is_recent": bool(getattr(song, "is_recent", False)),
        "is_top_track": bool(getattr(song, "is_top_track", False)),
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


def catalog_preference_scores(
    catalog: pd.DataFrame,
    profile: dict[str, Any],
    spec: QuerySpec | None = None,
    intent: PromptIntent | None = None,
) -> pd.Series:
    if catalog.empty:
        return pd.Series(dtype=float)
    prompt_terms: list[str] = []
    intent = intent or PromptIntent()
    if spec is not None:
        prompt_terms.extend(spec.genres)
        prompt_terms.extend(spec.moods)
        prompt_terms.extend(spec.seed_artists)
        if spec.seed_track:
            prompt_terms.append(spec.seed_track)
    prompt_terms.extend(intent.semantic_terms)

    score_frame = discovery_score_frame(
        catalog,
        profile,
        prompt_terms,
        AUDIO_FEATURES,
        weights=DEFAULT_DISCOVERY_WEIGHTS,
    )
    if score_frame.empty:
        return pd.Series(0.0, index=catalog.index, dtype=float)

    artist_col = catalog["artists"].fillna("")
    genre_col = catalog["track_genre"].fillna("").str.lower()
    score = score_frame["discovery_score"].copy()

    directed_prompt = prompt_has_clear_direction(spec)
    if directed_prompt and spec is not None:
        if spec.seed_artists:
            artist_bonus = pd.Series(0.0, index=catalog.index, dtype=float)
            for artist in spec.seed_artists[:3]:
                artist_bonus += artist_col.str.contains(re.escape(artist), case=False, na=False).astype(float) * 0.18
            score += artist_bonus
        if spec.genres:
            genre_bonus = pd.Series(0.0, index=catalog.index, dtype=float)
            for genre in spec.genres[:4]:
                genre_bonus += genre_col.str.contains(re.escape(normalize_affinity_label(genre)), case=False, na=False).astype(float) * 0.10
            score += genre_bonus

    us_genre_bonus = pd.Series(0.0, index=catalog.index, dtype=float)
    for token in US_HIT_GENRE_TOKENS:
        us_genre_bonus += genre_col.str.contains(re.escape(token), case=False, na=False).astype(float) * 0.10

    world_penalty = pd.Series(0.0, index=catalog.index, dtype=float)
    for token in DEPRIORITIZED_WORLD_GENRE_TOKENS:
        world_penalty += genre_col.str.contains(re.escape(token), case=False, na=False).astype(float) * 0.12

    if intent.excluded_artists:
        excluded_penalty = pd.Series(0.0, index=catalog.index, dtype=float)
        for artist in intent.excluded_artists[:3]:
            excluded_penalty += artist_col.str.contains(re.escape(artist), case=False, na=False).astype(float) * 0.5
        score -= excluded_penalty

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
        return 0.04
    if spec.moods or spec.constraints:
        return 0.06
    if spec.seed_artists or spec.seed_track:
        return 0.09
    return 0.14


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


def refresh_session_external_bias(session: DJSession) -> None:
    if session is None or getattr(session, "pool", None) is None:
        return
    catalog = getattr(session.pool, "_df", None)
    if catalog is None or len(catalog) == 0:
        return
    session.pool.set_external_bias(
        catalog_preference_scores(
            catalog,
            current_taste_profile(),
            session.spec,
            getattr(session, "prompt_intent", None),
        ).to_numpy(dtype=float)
    )


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
    saved_tracks = spotify_paginated_items("https://api.spotify.com/v1/me/tracks", token, {"limit": "50"}, pages=10)
    recent_tracks = spotify_paginated_items("https://api.spotify.com/v1/me/player/recently-played", token, {"limit": "50"}, pages=3)

    if not any((short_artists, medium_artists, long_artists, short_tracks, medium_tracks, long_tracks, saved_tracks, recent_tracks)):
        return None

    artist_weights: list[tuple[str, float]] = []
    genre_weights: list[tuple[str, float]] = []
    track_weights: list[tuple[str, float]] = []
    liked_payloads: list[dict[str, Any]] = []
    saved_track_ids: set[str] = set()
    top_track_ids: set[str] = set()
    recent_track_ids: set[str] = set()
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
            if liked:
                top_track_ids.add(track_id)
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

    ingest_artists(short_artists, artist_boost=2.1, genre_boost=1.4)
    ingest_artists(medium_artists, artist_boost=1.6, genre_boost=1.1)
    ingest_artists(long_artists, artist_boost=1.2, genre_boost=0.9)

    for payload, boost in ((short_tracks, 1.9), (medium_tracks, 1.5), (long_tracks, 1.2)):
        for track in (payload or {}).get("items", []) or []:
            ingest_track_item(track, boost=boost, liked=True)

    for item in saved_tracks:
        track = item.get("track") or {}
        if isinstance(track, dict):
            track_id = str(track.get("id", "") or "")
            if track_id:
                saved_track_ids.add(track_id)
            ingest_track_item(track, boost=1.1, liked=True)

    for item in recent_tracks:
        track = item.get("track") or {}
        if isinstance(track, dict):
            track_id = str(track.get("id", "") or "")
            if track_id:
                recent_track_ids.add(track_id)
            ingest_track_item(track, boost=0.8, liked=False)

    if not genre_weights:
        # Collect unique artist IDs from track payloads and fetch their genres via Spotify API
        artist_id_to_boost: dict[str, float] = {}
        track_artist_map: dict[str, list[str]] = {}  # track_id -> [artist_id, ...]
        for track, boost in track_payloads:
            track_id = str(track.get("id", "") or "")
            t_artist_ids = []
            for a in track.get("artists") or []:
                aid = str(a.get("id", "") or "")
                if aid:
                    t_artist_ids.append(aid)
                    if aid not in artist_id_to_boost or artist_id_to_boost[aid] < boost:
                        artist_id_to_boost[aid] = boost
            if track_id and t_artist_ids:
                track_artist_map[track_id] = t_artist_ids

        artist_genres: dict[str, list[str]] = {}  # artist_id -> genres
        all_artist_ids = list(artist_id_to_boost.keys())
        for chunk_start in range(0, len(all_artist_ids), 50):
            chunk = all_artist_ids[chunk_start: chunk_start + 50]
            resp = spotify_api_get(
                "https://api.spotify.com/v1/artists",
                token,
                {"ids": ",".join(chunk)},
            )
            for artist_obj in (resp or {}).get("artists") or []:
                if not isinstance(artist_obj, dict):
                    continue
                aid = str(artist_obj.get("id", "") or "")
                genres = [str(g) for g in (artist_obj.get("genres") or []) if g]
                if aid and genres:
                    artist_genres[aid] = genres

        for track, boost in track_payloads:
            track_id = str(track.get("id", "") or "")
            t_artist_ids = track_artist_map.get(track_id, [])
            track_genres: list[str] = []
            for aid in t_artist_ids:
                track_genres.extend(artist_genres.get(aid, []))
            for genre in track_genres[:2]:
                genre_weights.append((genre, boost * 0.9))
            if track_genres and track_id:
                for liked in liked_payloads:
                    if liked.get("track_id") == track_id and not liked.get("genre"):
                        liked["genre"] = track_genres[0]

    for artist_name, weight in artist_weights:
        bump_affinity(state.setdefault("artist_affinity", {}), [artist_name], weight)
        update_beta_bucket(state.setdefault("artist_posterior", {}), [artist_name], liked=True, amount=max(0.12, weight * 0.12), decay=0.999)
    for genre_name, weight in genre_weights:
        bump_affinity(state.setdefault("genre_affinity", {}), [genre_name], weight)
        update_beta_bucket(state.setdefault("genre_posterior", {}), [genre_name], liked=True, amount=max(0.10, weight * 0.10), decay=0.999)
    for track_name, weight in track_weights:
        bump_affinity(state.setdefault("track_affinity", {}), [track_name], weight)
        update_beta_bucket(state.setdefault("track_posterior", {}), [track_name], liked=True, amount=max(0.06, weight * 0.06), decay=0.999)

    existing = state.setdefault("liked_songs", [])
    state["liked_songs"] = merge_liked_payloads(existing, liked_payloads, limit=48)
    state["spotify_saved_track_ids"] = sorted(set(state.get("spotify_saved_track_ids", [])) | saved_track_ids)
    state["spotify_top_track_ids"] = sorted(set(state.get("spotify_top_track_ids", [])) | top_track_ids)
    state["spotify_recent_track_ids"] = sorted(set(state.get("spotify_recent_track_ids", [])) | recent_track_ids)
    state["spotify_user_seeded"] = True
    state["spotify_user_summary"] = {
        "artists": top_affinity_items(state.get("artist_affinity", {}), limit=5),
        "genres": top_affinity_items(state.get("genre_affinity", {}), limit=5),
        "tracks": top_affinity_items(state.get("track_affinity", {}), limit=5),
    }
    novelty_bucket = {"novelty": state.get("novelty_posterior", {"alpha": 6.5, "beta": 3.5})}
    update_beta_bucket(novelty_bucket, ["novelty"], liked=True, amount=0.55, decay=0.999)
    state["novelty_posterior"] = novelty_bucket["novelty"]
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
    intent = parse_prompt_intent(prompt, spec, build_prompt_context())
    infer_preferences_from_message(prompt, spec)
    state = current_taste_profile()
    prompt_history = list(state.get("session_prompt_history", []))
    prompt_history.insert(0, prompt)
    state["session_prompt_history"] = prompt_history[:24]
    save_ui_state(state)
    st.session_state["seen_track_ids"] = set()
    st.session_state["seen_track_signatures"] = set()
    st.session_state["completed_song_count"] = 0
    st.session_state["completed_song_keys"] = set()
    session, taste_note = build_session_from_spec(
        spec,
        playlist_length,
        prompt_intent=intent,
        excluded_track_ids=set(),
        excluded_track_signatures=set(),
    )
    spec = session.spec  # use blended spec from build_session_from_spec
    st.session_state["dj_session"] = session
    st.session_state["session_finished"] = session._current_song is None
    st.session_state["last_feedback"] = None
    st.session_state["chat_messages"] = []
    st.session_state["latest_model_update"] = f"Session initialized from: {prompt}"
    st.session_state["playback_song"] = None
    st.session_state["playback_scored"] = False
    st.session_state["back_song"] = None
    st.session_state["showing_back_song"] = False
    st.session_state["welcome_greeted"] = True
    add_chat_message("user", prompt)
    response = compose_assistant_message(
        f"**You asked for:** {prompt}",
        humanize_spotify_note(spotify_note),
        humanize_taste_note(taste_note),
        summarize_spec(spec, session),
    )
    add_chat_message("assistant", response)
    schedule_speech(plain_speech_text(summarize_spec(spec, session)), f"query-{abs(hash((prompt, session._current_song.track_id if session._current_song is not None else ''))) }")


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


def _build_refinement_with_fallback(
    base_merged_spec: "QuerySpec",
    refinement_intent: "PromptIntent",
    current_session: "DJSession",
    prior_feedback: list,
    excluded_ids: set,
    excluded_sigs: set,
) -> "tuple[DJSession | None, str | None, str]":
    """Try increasingly relaxed versions of the merged spec until we get a non-empty pool.

    Returns ``(rebuilt_session, taste_note, relaxation_note)`` where
    *relaxation_note* describes what was relaxed (empty string if strict pass worked).
    """
    playlist_length = current_session.playlist_length

    # Build a widened intent for fallback stages
    wider_intent = PromptIntent(**vars(refinement_intent))
    wider_intent.exploration_weight = min(0.96, float(refinement_intent.exploration_weight) + 0.12)
    wider_intent.novelty_target = min(0.96, float(refinement_intent.novelty_target) + 0.12)
    wider_intent.direct_match_weight = max(0.22, float(refinement_intent.direct_match_weight) - 0.08)

    fallback_chain: list[tuple["QuerySpec", "PromptIntent", str]] = [
        # Stage 1 — strict as requested
        (base_merged_spec, refinement_intent, ""),
    ]

    # Stage 2 — widen exploration/novelty while keeping all filters
    fallback_chain.append((clone_spec(base_merged_spec), wider_intent, ""))

    # Stage 3 — only the first genre (drop secondary genre constraints)
    if base_merged_spec.genres and len(base_merged_spec.genres) > 1:
        relaxed_genres = clone_spec(base_merged_spec)
        relaxed_genres.genres = list(base_merged_spec.genres[:1])
        relaxed_genres.spotify_search_queries = relaxed_genres.to_spotify_search_queries()
        fallback_chain.append((relaxed_genres, wider_intent, ""))

    # Stage 4 — keep genres but drop audio-feature constraints so Spotify
    #           returns a broader candidate pool
    if base_merged_spec.constraints:
        no_constraints = clone_spec(base_merged_spec)
        no_constraints.constraints = {}
        no_constraints.spotify_search_queries = no_constraints.to_spotify_search_queries()
        fallback_chain.append(
            (no_constraints, wider_intent, "I loosened the audio-feature filters to open up the pool.")
        )

    # Stage 5 — broad: keep seed artists/tracks but drop all genre + constraint
    #           filters so we always find *something*
    broad_spec = clone_spec(base_merged_spec)
    broad_spec.genres = []
    broad_spec.constraints = {}
    broad_spec.spotify_search_queries = broad_spec.to_spotify_search_queries()
    fallback_chain.append(
        (broad_spec, wider_intent, "I dropped the genre filter too so I could find enough tracks.")
    )

    for candidate_spec, candidate_intent, relax_note in fallback_chain:
        session, taste_note = build_session_from_spec(
            candidate_spec,
            playlist_length,
            prompt_intent=candidate_intent,
            prior_feedback=prior_feedback,
            excluded_track_ids=excluded_ids,
            excluded_track_signatures=excluded_sigs,
        )
        count = getattr(session, "initial_candidate_count", session.pool.n_available)
        if count > 0:
            return session, taste_note, relax_note

    return None, None, ""


def apply_refinement(message: str) -> None:
    current_session = st.session_state["dj_session"]
    previous_spec = clone_spec(current_session.spec)
    refinement = enrich_spec_from_prompt(message, get_parser().parse(message))
    infer_preferences_from_message(message, refinement)
    state = current_taste_profile()
    prompt_history = list(state.get("session_prompt_history", []))
    prompt_history.insert(0, message)
    state["session_prompt_history"] = prompt_history[:24]
    save_ui_state(state)
    spotify_note = sync_spotify_user_preferences()
    merged_spec = merge_specs(previous_spec, refinement, message)
    refinement_intent = parse_prompt_intent(message, merged_spec, build_prompt_context())
    prior_feedback = session_feedback_history(current_session)
    excluded_ids = set(st.session_state.get("seen_track_ids", set()))
    excluded_sigs = set(st.session_state.get("seen_track_signatures", set()))

    rebuilt_session, taste_note, relax_note = _build_refinement_with_fallback(
        merged_spec,
        refinement_intent,
        current_session,
        prior_feedback,
        excluded_ids,
        excluded_sigs,
    )

    if rebuilt_session is None:
        # Absolute last resort: keep the previous session but acknowledge the request
        add_chat_message("user", message)
        add_chat_message(
            "assistant",
            "I'm having trouble pulling enough tracks for that exact request right now — I'll keep the current set going and steer toward that vibe gradually. Try nudging me again or give me a broader direction.",
        )
        return

    merged_spec = rebuilt_session.spec  # use blended spec

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
    relax_section = f"**Note:** {relax_note}" if relax_note else None
    assistant_reply = compose_assistant_message(
        f"**I'm reshaping the set:** {change_summary}",
        relax_section,
        humanize_spotify_note(spotify_note),
        humanize_taste_note(taste_note),
        f"**I'm keeping your earlier reactions in the mix too:** {len(prior_feedback)} prior decisions are still informing this.",
        summarize_spec(merged_spec, rebuilt_session),
    )
    add_chat_message("assistant", assistant_reply)
    schedule_speech(plain_speech_text(summarize_spec(merged_spec, rebuilt_session)), f"refine-{abs(hash((message, rebuilt_session._current_song.track_id if rebuilt_session._current_song is not None else '')))}")


GENRE_TRANSITIONS = {
    # Pop and adjacent
    "pop": ["funk", "soul", "r&b", "dance pop", "disco"],
    "dance pop": ["disco", "funk", "house", "pop"],
    "synth pop": ["new wave", "electro pop", "indie pop", "dance pop"],
    "electro pop": ["synth pop", "dance pop", "house", "new wave"],
    "indie pop": ["indie", "alternative", "folk", "dream pop"],
    "dream pop": ["shoegaze", "indie pop", "alternative", "ambient"],
    # Soul, funk, R&B
    "soul": ["funk", "r&b", "neo soul", "jazz", "gospel"],
    "funk": ["disco", "soul", "r&b", "boogie", "dance pop"],
    "r&b": ["soul", "funk", "neo soul", "hip hop", "jazz"],
    "neo soul": ["soul", "r&b", "jazz", "hip hop", "funk"],
    "boogie": ["funk", "disco", "soul", "dance pop"],
    # Disco and dance
    "disco": ["funk", "house", "boogie", "soul", "dance pop"],
    "house": ["disco", "dance", "soul", "techno", "deep house"],
    "deep house": ["house", "disco", "soul", "afrobeats"],
    "dance": ["house", "disco", "funk", "dance pop"],
    # Hip hop and related
    "hip hop": ["r&b", "neo soul", "house", "trap", "funk"],
    "trap": ["hip hop", "r&b", "pop rap", "cloud rap"],
    "pop rap": ["hip hop", "trap", "dance pop", "r&b"],
    # Rock and adjacent
    "rock": ["indie", "new wave", "alternative", "classic rock"],
    "classic rock": ["blues rock", "hard rock", "southern rock", "rock"],
    "indie": ["indie pop", "alternative", "folk", "dream pop"],
    "alternative": ["indie", "rock", "grunge", "post punk"],
    "new wave": ["synth pop", "post punk", "electro pop", "indie"],
    "post punk": ["new wave", "alternative", "gothic rock", "indie"],
    # Jazz and adjacent
    "jazz": ["neo soul", "soul", "funk", "bossa nova", "blues"],
    "blues": ["soul", "r&b", "funk", "classic rock", "jazz"],
    "blues rock": ["classic rock", "rock", "blues", "southern rock"],
    "bossa nova": ["jazz", "latin", "soul", "afrobeats"],
    # Electronic
    "trance": ["melodic techno", "progressive house", "dance", "house"],
    "techno": ["house", "trance", "minimal techno", "industrial"],
    "ambient": ["dream pop", "shoegaze", "focus", "new age"],
    "lofi": ["hip hop", "jazz", "neo soul", "ambient"],
    # World and folk
    "afrobeats": ["afropop", "dancehall", "funk", "r&b", "deep house"],
    "afropop": ["afrobeats", "pop", "soul", "funk"],
    "latin": ["bossa nova", "salsa", "reggaeton", "pop"],
    "reggaeton": ["latin", "trap", "dance pop", "r&b"],
    "folk": ["indie", "alternative", "country", "singer songwriter"],
    "country": ["folk", "country pop", "americana", "southern rock"],
    "americana": ["folk", "country", "blues", "rock"],
    # Misc
    "gospel": ["soul", "r&b", "funk", "neo soul"],
    "shoegaze": ["dream pop", "alternative", "indie", "post punk"],
    "metal": ["hard rock", "alternative", "rock", "classic rock"],
    "hard rock": ["metal", "classic rock", "rock", "blues rock"],
}

# Maps well-known artist names (lowercased) to their inferred genre and
# a chain of musically adjacent pivot genres.  Used when a session is
# seeded by artist name but has no explicit genres set in the spec.
ARTIST_PIVOT_GENRES: dict[str, list[str]] = {
    # Pop/soul crossovers
    "michael jackson": ["funk", "disco", "soul", "r&b", "boogie"],
    "prince": ["funk", "r&b", "soul", "disco", "dance pop"],
    "janet jackson": ["funk", "r&b", "dance pop", "soul", "disco"],
    "stevie wonder": ["soul", "funk", "r&b", "neo soul", "jazz"],
    "whitney houston": ["r&b", "soul", "dance pop", "gospel"],
    "mariah carey": ["r&b", "soul", "pop", "gospel"],
    "beyonce": ["r&b", "pop", "soul", "dance pop", "funk"],
    "tina turner": ["soul", "rock", "r&b", "funk"],
    # Funk & disco
    "earth wind fire": ["soul", "funk", "r&b", "disco", "gospel"],
    "earth, wind & fire": ["soul", "funk", "r&b", "disco", "gospel"],
    "parliament": ["funk", "soul", "r&b", "disco"],
    "funkadelic": ["funk", "soul", "rock", "r&b"],
    "james brown": ["funk", "soul", "r&b", "gospel"],
    "sly and the family stone": ["funk", "soul", "rock", "r&b"],
    "chic": ["disco", "funk", "soul", "dance pop"],
    "kool and the gang": ["funk", "soul", "r&b", "disco"],
    "the gap band": ["funk", "soul", "r&b", "dance pop"],
    # R&B/neo-soul
    "d'angelo": ["neo soul", "soul", "funk", "r&b"],
    "erykah badu": ["neo soul", "soul", "jazz", "r&b"],
    "lauryn hill": ["neo soul", "r&b", "hip hop", "soul"],
    "john legend": ["r&b", "soul", "pop", "neo soul"],
    "usher": ["r&b", "dance pop", "pop", "soul"],
    # Hip hop
    "kendrick lamar": ["hip hop", "r&b", "neo soul", "jazz"],
    "drake": ["hip hop", "r&b", "trap", "pop rap"],
    "j. cole": ["hip hop", "r&b", "neo soul", "jazz"],
    "frank ocean": ["r&b", "neo soul", "indie pop", "soul"],
    # Classic rock / oldies
    "the beatles": ["rock", "folk", "classic rock", "pop"],
    "led zeppelin": ["classic rock", "blues rock", "hard rock", "folk"],
    "rolling stones": ["classic rock", "blues rock", "rock", "soul"],
    "david bowie": ["rock", "new wave", "synth pop", "glam rock"],
    "fleetwood mac": ["classic rock", "folk", "soft rock", "indie"],
    # Electronic / dance
    "daft punk": ["house", "disco", "electro", "funk"],
    "the weeknd": ["r&b", "synth pop", "pop", "neo soul"],
    "post malone": ["pop rap", "hip hop", "trap", "rock"],
    # Pop
    "taylor swift": ["indie pop", "pop", "country pop", "folk"],
    "billie eilish": ["indie pop", "alternative", "electro pop", "dream pop"],
    "bruno mars": ["funk", "soul", "pop", "r&b", "disco"],
    "the 1975": ["indie pop", "synth pop", "alternative", "new wave"],
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
    current_genres = {g.lower() for g in spec.genres}

    # 1. Try genre-based transitions using expanded GENRE_TRANSITIONS table
    for current_genre in spec.genres:
        options = GENRE_TRANSITIONS.get(current_genre.lower(), [])
        for candidate in options:
            if candidate.lower() not in current_genres:
                return candidate

    # 2. Try artist-based pivot: look up seed_artists in ARTIST_PIVOT_GENRES
    for artist in getattr(spec, "seed_artists", []):
        artist_key = artist.lower().strip()
        # Try exact match first, then partial
        artist_pivots = ARTIST_PIVOT_GENRES.get(artist_key)
        if not artist_pivots:
            for key, pivots in ARTIST_PIVOT_GENRES.items():
                if key in artist_key or artist_key in key:
                    artist_pivots = pivots
                    break
        if artist_pivots:
            for candidate in artist_pivots:
                if candidate.lower() not in current_genres:
                    return candidate

    # 3. Fall back to user's taste profile genre affinities
    preferred = top_affinity_items(current_taste_profile().get("genre_affinity", {}), limit=4)
    for candidate in preferred:
        if candidate.lower() not in current_genres:
            return candidate

    # 4. Generic fallback list
    fallback = ["neo soul", "funk", "indie pop", "house", "jazz", "soul", "folk"]
    for candidate in fallback:
        if candidate.lower() not in current_genres:
            return candidate
    return spec.genres[0] if spec.genres else None


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


def _pivot_transition_speech(spec: "QuerySpec", pivot_genre: str) -> str:
    """Generate a contextual DJ narration line for a genre pivot.

    Describes the *from* context (seed artists or current genre) and the *to*
    destination so the transition feels intentional, not random.
    """
    # Describe what we're moving away from
    from_parts: list[str] = []
    if getattr(spec, "seed_artists", []):
        artist_names = ", ".join(spec.seed_artists[:2])
        from_parts.append(artist_names)
    if spec.genres:
        from_parts.append(spec.genres[0])

    if from_parts:
        from_label = " / ".join(from_parts)
        return (
            f"Alright, we've been deep in the {from_label} lane — now I'm shifting the energy "
            f"and taking you into some {pivot_genre} territory. Same thread, new groove."
        )
    return (
        f"DJ Bayes pivot. I'm steering this set into a {pivot_genre} lane for the next stretch "
        f"while keeping what I've learned about your taste intact."
    )


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
        pivot_speech = _pivot_transition_speech(spec, pivot_genre)
        from_label = (spec.seed_artists[0] if getattr(spec, "seed_artists", []) else None) or (spec.genres[0] if spec.genres else "the current sound")
        add_route(
            "genre_pivot",
            f"Moving from **{from_label}** into **{pivot_genre}** territory.",
            "You have enough posterior signal here, so this is a controlled genre pivot rather than a reset.",
            pivot_speech,
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


def ensure_dj_greeting() -> None:
    if st.session_state.get("welcome_greeted"):
        return
    greeting = compose_assistant_message(
        "**Hey, I'm DJ Bayes.** Good to have you here.",
        "**Throw me a vibe, an artist, or a moment:** I'll turn it into a live set that feels personal, playful, and easy to steer.",
        "**House rule:** every five songs or so, I'll steer the set into a new but related direction — keeping things fresh without losing the thread.",
    )
    add_chat_message("assistant", greeting)
    schedule_speech(
        "Hey now, I'm DJ Bayes. Slide me a vibe, an artist, or a moment, and I'll get the room moving for you.",
        "welcome-greeting",
    )
    st.session_state["latest_model_update"] = "DJ Bayes is live and ready to spin."
    st.session_state["welcome_greeted"] = True


def maybe_trigger_dj_interlude() -> bool:
    session = st.session_state["dj_session"]
    completed_songs = int(st.session_state.get("completed_song_count", 0))
    if completed_songs == 0 or completed_songs % 5 != 0:
        return False
    if st.session_state.get("last_transition_round") == completed_songs:
        return False

    prior_feedback = session_feedback_history(session)
    route_history = st.session_state.get("recent_intervention_routes", [])
    routes = build_intervention_routes(session.spec, completed_songs)
    if route_history:
        filtered = [route for route in routes if route["name"] not in route_history[-2:]]
        if filtered:
            routes = filtered

    chosen_route: dict[str, Any] | None = None
    rebuilt: DJSession | None = None
    for route in routes:
        candidate_spec = route["builder"](session.spec)
        candidate_session, _ = build_session_from_spec(
            candidate_spec,
            session.playlist_length,
            prior_feedback=prior_feedback,
            excluded_track_ids=set(st.session_state.get("seen_track_ids", set())),
            excluded_track_signatures=set(st.session_state.get("seen_track_signatures", set())),
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

    st.session_state["last_transition_round"] = completed_songs
    transition_count = completed_songs
    if chosen_route is not None and rebuilt is not None and rebuilt._current_song is not None:
        message = compose_assistant_message(
            f"**DJ Bayes — set pivot after {transition_count} songs:** {chosen_route['headline']}",
            f"**Why this route:** {chosen_route['reason']}",
            f"**Next up:** {rebuilt._current_song.track_name} by {rebuilt._current_song.artists}",
        )
        latest_update = f"Auto-intervention after {transition_count} songs: {chosen_route['headline'].replace('**', '')}"
        speech_text = str(chosen_route["speech"])
    else:
        ensure_current_song(session)
        fallback_genre = next_transition_genre(session.spec) or "a new lane"
        fallback_speech = _pivot_transition_speech(session.spec, fallback_genre)
        message = compose_assistant_message(
            f"**DJ Bayes — set pivot after {transition_count} songs:** I'm nudging the set toward **{fallback_genre}**.",
            "**Why this route:** I couldn't find a stronger alternate detour with enough songs, so I kept the pivot simple.",
            f"**Next up:** {session._current_song.track_name} by {session._current_song.artists}" if session._current_song is not None else None,
        )
        latest_update = f"Auto-intervention after {transition_count} songs toward {fallback_genre}."
        speech_text = fallback_speech
    add_chat_message("assistant", message)
    st.session_state["latest_model_update"] = latest_update
    schedule_speech(speech_text, f"transition-{transition_count}")
    return True


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
        const topWindow = window.parent && window.parent !== window ? window.parent : window;
        const storage = topWindow.sessionStorage || window.sessionStorage;
        const synth = topWindow.speechSynthesis || window.speechSynthesis;
        function pickVoice() {{
            const voices = synth.getVoices() || [];
            const englishVoices = voices.filter((voice) => /en[-_]/i.test(voice.lang || ""));
            const preferred = [
                /daniel/i,
                /alex/i,
                /aaron/i,
                /fred/i,
                /reed/i,
                /david/i,
                /tom/i,
                /bruce/i,
                /male/i
            ];
            for (const pattern of preferred) {{
                const match = englishVoices.find((voice) => pattern.test(`${{voice.name}} ${{voice.voiceURI || ""}}`));
                if (match) {{
                    return match;
                }}
            }}
            return englishVoices.find((voice) => /en-us/i.test(voice.lang || "")) || englishVoices[0] || voices[0] || null;
        }}
        function speakNow() {{
            storage.setItem(storeKey, key);
            storage.setItem(pendingKey, "1");
            synth.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            const selectedVoice = pickVoice();
            if (selectedVoice) {{
                utterance.voice = selectedVoice;
            }}
            utterance.rate = 0.96;
            utterance.pitch = 0.84;
            utterance.volume = 1.0;
            utterance.lang = "en-US";
            utterance.onend = () => {{
                storage.removeItem(pendingKey);
            }};
            utterance.onerror = () => {{
                storage.removeItem(pendingKey);
            }};
            synth.speak(utterance);
        }}
        if (storage.getItem(storeKey) !== key) {{
            const voices = synth.getVoices() || [];
            if (!voices.length) {{
                const handleVoicesChanged = () => {{
                    synth.removeEventListener("voiceschanged", handleVoicesChanged);
                    speakNow();
                }};
                synth.addEventListener("voiceschanged", handleVoicesChanged, {{ once: true }});
                window.setTimeout(() => {{
                    if (storage.getItem(storeKey) !== key) {{
                        speakNow();
                    }}
                }}, 350);
            }} else {{
                speakNow();
            }}
        }} else {{
            storage.removeItem(pendingKey);
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
        base = "This sits right in your pocket and still feels like a worthwhile pull."
    elif score >= 0.64:
        base = "This lines up with your taste while keeping some discovery in the mix."
    elif score >= 0.46:
        base = "This leans exploratory, but there is enough overlap for it to land."
    else:
        base = "This is a bolder discovery swing, but it still shares DNA with your profile."
    if not song.is_saved and not song.is_top_track:
        base += " It is also newer to you than the average pick."
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
    queued_prompt = str(st.session_state.get("queued_prompt", "") or "").strip()
    if queued_prompt:
        st.markdown(
            f"""
            <div class="chat-bubble user">
                <div class="chat-role">You</div>
                <div class="chat-body">{render_message_html(queued_prompt)}</div>
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
        mark_song_completed(song)
        if not maybe_trigger_dj_interlude():
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
                const stateKey = "dj-bayes-audio-state";
                const currentTrackId = {json.dumps(str(track_id))};
                const topWindow = window.parent && window.parent !== window ? window.parent : window;
                const storage = topWindow.sessionStorage || window.sessionStorage;
                const synth = topWindow.speechSynthesis || window.speechSynthesis;
                function loadState() {{
                    try {{
                        return JSON.parse(storage.getItem(stateKey) || "{{}}");
                    }} catch (err) {{
                        return {{}};
                    }}
                }}
                function saveState(next) {{
                    try {{
                        storage.setItem(stateKey, JSON.stringify(next));
                    }} catch (err) {{}}
                }}
                async function waitForVoice() {{
                    if (!deferForVoice) {{
                        return;
                    }}
                    const pendingKey = "dj-bayes-pending-speech";
                    while (storage.getItem(pendingKey) === "1" || synth.speaking) {{
                        await new Promise((resolve) => setTimeout(resolve, 250));
                    }}
                }}
                async function startPlayback() {{
                    const savedState = loadState();
                    if (savedState.trackId === currentTrackId && Number.isFinite(savedState.position)) {{
                        try {{
                            audio.currentTime = Math.max(0, Number(savedState.position) || 0);
                        }} catch (err) {{}}
                    }}
                    try {{
                        await waitForVoice();
                        if (savedState.trackId === currentTrackId && savedState.paused) {{
                            return;
                        }}
                        const maybePromise = audio.play();
                        if (maybePromise) {{
                            await maybePromise;
                        }}
                    }} catch (err) {{}}
                }}
                function persistAudioState() {{
                    saveState({{
                        trackId: currentTrackId,
                        position: Number(audio.currentTime || 0),
                        paused: Boolean(audio.paused),
                    }});
                }}
                audio.addEventListener("timeupdate", persistAudioState);
                audio.addEventListener("pause", persistAudioState);
                audio.addEventListener("play", persistAudioState);
                startPlayback();
                audio.addEventListener("ended", () => {{
                    saveState({{ trackId: currentTrackId, position: 0, paused: true }});
                    goNext();
                }});
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
                const stateKey = "dj-bayes-spotify-state";
                const currentTrackId = {json.dumps(str(track_id))};
                const topWindow = window.parent && window.parent !== window ? window.parent : window;
                const storage = topWindow.sessionStorage || window.sessionStorage;
                const synth = topWindow.speechSynthesis || window.speechSynthesis;
                function loadState() {{
                    try {{
                        return JSON.parse(storage.getItem(stateKey) || "{{}}");
                    }} catch (err) {{
                        return {{}};
                    }}
                }}
                function saveState(next) {{
                    try {{
                        storage.setItem(stateKey, JSON.stringify(next));
                    }} catch (err) {{}}
                }}
                async function waitForVoice() {{
                    if (!deferForVoice) {{
                        return;
                    }}
                    const pendingKey = "dj-bayes-pending-speech";
                    while (storage.getItem(pendingKey) === "1" || synth.speaking) {{
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
                        const savedState = loadState();
                        try {{
                            await waitForVoice();
                            if (savedState.trackId === currentTrackId && Number.isFinite(savedState.position) && typeof EmbedController.seek === "function") {{
                                try {{
                                    EmbedController.seek(Number(savedState.position) || 0);
                                }} catch (err) {{}}
                            }}
                            if (!(savedState.trackId === currentTrackId && savedState.paused)) {{
                                EmbedController.play();
                            }}
                        }} catch (err) {{}}
                        let autoAdvanced = false;
                        EmbedController.addListener("playback_update", (event) => {{
                            const data = event && event.data ? event.data : null;
                            if (!data || autoAdvanced) {{
                                return;
                            }}
                            saveState({{
                                trackId: currentTrackId,
                                position: Number(data.position || 0),
                                paused: Boolean(data.isPaused),
                            }});
                            const duration = Number(data.duration || 0);
                            const position = Number(data.position || 0);
                            if (duration > 0 && position >= duration - 1000) {{
                                autoAdvanced = true;
                                saveState({{ trackId: currentTrackId, position: 0, paused: true }});
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
        current_feedback = current_feedback_for_song(session, song)
        feedback_cols = st.columns(2)
        with feedback_cols[0]:
            save_disabled = (
                playback_song is not None
                or st.session_state.get("showing_back_song", False)
                or current_feedback is not None
            )
            save_label = "Like this recommendation?"
            if st.button(save_label, width="stretch", type="primary", disabled=save_disabled):
                clear_pending_reaction()
                apply_positive_feedback(
                    session,
                    song,
                    "Liked this recommendation.",
                )
        with feedback_cols[1]:
            dislike_disabled = (
                playback_song is not None
                or st.session_state.get("showing_back_song", False)
                or current_feedback is not None
            )
            if st.button("Don't like this recommendation", width="stretch", disabled=dislike_disabled):
                clear_pending_reaction()
                apply_negative_feedback(session, song, "Didn't like this recommendation.")

        if playback_song is not None:
            st.caption("Liked songs stay visible until you move to the next recommendation.")
        elif current_feedback == "play":
            st.caption("You liked this recommendation. DJ Bayes already updated the model, and this song will keep playing until you move on.")
        elif current_feedback == "skip":
            st.caption("You marked this recommendation as a miss. DJ Bayes already updated the model, and the current song will keep playing until you move on.")
        elif saved_on_spotify is True:
            st.caption("This track is already in your Spotify library. You can still like or dislike it here so DJ Bayes learns how it fits this session.")
        elif st.session_state.get("showing_back_song", False):
            st.caption("Back is listen-only. Return to the live stream to keep training the model.")
        else:
            st.caption("Use these buttons to train DJ Bayes directly. Spotify library status is just a prior signal, not the final verdict.")

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
                        mark_song_completed(song)
                        session.advance_without_feedback()
                        if not maybe_trigger_dj_interlude():
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
            Spotify is connected and DJ Bayes is on deck. Ask for a vibe, scene, artist, or moment and the stream will start from your listening history.
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

    if st.session_state["dj_session"] is None:
        ensure_dj_greeting()

    session = st.session_state["dj_session"]
    if session is None:
        render_empty_workspace()
        render_conversation()
    else:
        render_current_track(session)
        render_conversation()
        render_posterior_panels(session)
        render_latest_update()

    follow_up = st.chat_input("Ask DJ Bayes for a vibe, mood, or scene")
    if follow_up:
        st.session_state["queued_prompt"] = follow_up
        st.rerun()

    queued_prompt = str(st.session_state.get("queued_prompt", "") or "").strip()
    if queued_prompt:
        st.session_state["queued_prompt"] = ""
        with st.spinner("Updating DJ Bayes..."):
            if st.session_state["dj_session"] is None:
                start_session(queued_prompt, st.session_state["playlist_length"])
            elif message_starts_new_request(queued_prompt):
                start_session(queued_prompt, st.session_state["playlist_length"])
            else:
                apply_refinement(queued_prompt)
        st.rerun()

    render_voice_interlude()


if __name__ == "__main__":
    main()
