from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from music_query_parser.parser import QuerySpec


ENERGY_MAP = {
    "low": 0.25,
    "soft": 0.28,
    "calm": 0.30,
    "smooth": 0.36,
    "chill": 0.38,
    "warm": 0.44,
    "upbeat": 0.68,
    "fun": 0.70,
    "hype": 0.86,
    "aggressive": 0.90,
    "hard": 0.92,
    "workout": 0.90,
}

VALENCE_MAP = {
    "dark": 0.22,
    "moody": 0.34,
    "smooth": 0.46,
    "warm": 0.60,
    "fun": 0.74,
    "funky": 0.78,
    "happy": 0.84,
}

TEMPO_MAP = {
    "late night": 92.0,
    "smooth": 96.0,
    "funky": 114.0,
    "fun": 118.0,
    "workout": 144.0,
    "hype": 140.0,
    "hard": 148.0,
}

ACTIVITY_TERMS = {
    "workout": "workout",
    "gym": "workout",
    "run": "workout",
    "pregame": "going out",
    "party": "going out",
    "going out": "going out",
    "study": "focus",
    "focus": "focus",
    "drive": "driving",
    "driving": "driving",
    "late night": "late night",
    "date night": "date night",
}

PRODUCTION_TERMS = {
    "bass-heavy": "bass heavy",
    "bass heavy": "bass heavy",
    "jazzy": "jazzy chords",
    "funky": "groovy rhythm section",
    "glossy": "polished production",
    "grimy": "grimy drums",
    "grittier": "grimy drums",
    "grimier": "grimy drums",
    "weird": "off-center production",
    "experimental": "experimental production",
    "atmospheric": "atmospheric layers",
}

VOCAL_TERMS = {
    "smooth": "smooth vocals",
    "raspy": "raspy vocals",
    "soulful": "soulful vocals",
    "melodic": "melodic vocals",
    "rapped": "rap-forward vocals",
    "vocal": "vocal-forward",
    "instrumental": "instrumental",
    "no vocals": "instrumental",
}

GENRE_HINTS = {
    "rap": "rap",
    "hip hop": "hip hop",
    "hip-hop": "hip hop",
    "trap": "trap",
    "drill": "drill",
    "rage": "rage rap",
    "pluggnb": "pluggnb",
    "r&b": "r&b",
    "alt r&b": "alt r&b",
    "neo soul": "neo soul",
    "soul": "soul",
    "funk": "funk",
    "disco": "disco",
    "house": "house",
    "afrobeats": "afrobeats",
    "jazz": "jazz",
    "indie": "indie",
    "indie rock": "indie rock",
    "indie pop": "indie pop",
    "shoegaze": "shoegaze",
    "dream pop": "dream pop",
    "rock": "rock",
    "hard rock": "hard rock",
    "punk": "punk",
    "metal": "metal",
    "country": "country",
    "folk": "folk",
    "ambient": "ambient",
    "electronic": "electronic",
    "dance": "dance pop",
    "dance pop": "dance pop",
    "techno": "techno",
    "jersey club": "jersey club",
    "detroit": "detroit rap",
    "west coast": "west coast hip hop",
    "uk rap": "uk rap",
    "gospel": "gospel",
}

MOOD_HINTS = {
    "rainy": "moody",
    "stormy": "dark",
    "sunny": "bright",
    "golden hour": "warm",
    "lowkey": "chill",
    "low-key": "chill",
    "floaty": "dreamy",
    "dreamy": "dreamy",
    "grimy": "dark",
    "gritty": "dark",
    "romantic": "romantic",
    "sexy": "smooth",
    "sultry": "smooth",
    "bouncy": "upbeat",
    "turnt": "hype",
    "lit": "hype",
    "peaceful": "calm",
    "focused": "calm",
    "melancholy": "moody",
    "melancholic": "moody",
    "sad": "moody",
}

INSTRUMENTATION_TERMS = {
    "guitar": "guitar-led",
    "piano": "piano-led",
    "strings": "string textures",
    "sax": "saxophone",
    "horns": "horn section",
    "808": "808-heavy",
    "drums": "drum-forward",
}

NOVELTY_HIGH = [
    "new",
    "fresh",
    "different",
    "unique",
    "weird",
    "experimental",
    "less mainstream",
    "underground",
    "deeper",
    "deeper cuts",
    "left field",
    "left-field",
    "more interesting",
    "unexpected",
]

NOVELTY_LOW = [
    "hits",
    "classics",
    "familiar",
    "well known",
    "well-known",
    "popular",
    "just the classics",
    "play the hits",
]

SIMILARITY_PHRASES = [
    "more like this",
    "same vibe",
    "like this but",
    "music like",
    "similar to",
    "in the style of",
]

REFERENCE_PHRASES = [
    "like",
    "similar to",
    "in the style of",
    "based on",
    "around",
]

FOLLOW_UP_PHRASES = [
    "more like this",
    "same vibe",
    "keep this",
    "less intense",
    "more upbeat",
    "less mainstream",
    "more underground",
    "make it",
]

ERA_PATTERNS = [
    (r"\b90s\b|\b1990s\b", "1990s"),
    (r"\b00s\b|\b2000s\b", "2000s"),
    (r"\b2010s\b", "2010s"),
    (r"\b2020s\b|\bnew release\b|\bnewer\b", "2020s"),
]


@dataclass
class PromptIntent:
    seed_artists: list[str] = field(default_factory=list)
    seed_tracks: list[str] = field(default_factory=list)
    excluded_artists: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    energy_target: float | None = None
    valence_target: float | None = None
    tempo_target: float | None = None
    novelty_target: float = 0.72
    familiarity_target: float = 0.28
    activity_context: list[str] = field(default_factory=list)
    era: str | None = None
    production_tags: list[str] = field(default_factory=list)
    vocal_tags: list[str] = field(default_factory=list)
    direct_match_weight: float = 0.45
    exploration_weight: float = 0.55
    similarity_mode: str = "blend"
    requested_single_artist: bool = False
    follow_up: bool = False
    semantic_terms: list[str] = field(default_factory=list)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _weighted_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def parse_prompt_intent(prompt: str, spec: QuerySpec, session_context: dict[str, Any] | None = None) -> PromptIntent:
    text = " ".join((prompt or "").split())
    lowered = text.lower()
    session_context = session_context or {}

    moods = _dedupe(list(spec.moods))
    genres = _dedupe(list(spec.genres))
    seed_artists = _dedupe(list(spec.seed_artists))
    seed_tracks = [spec.seed_track] if spec.seed_track else []

    for phrase, genre in GENRE_HINTS.items():
        if phrase in lowered:
            genres.append(genre)
    for phrase, mood in MOOD_HINTS.items():
        if phrase in lowered:
            moods.append(mood)

    novelty_score = 0.72
    for phrase in NOVELTY_HIGH:
        if phrase in lowered:
            novelty_score += 0.10
    for phrase in NOVELTY_LOW:
        if phrase in lowered:
            novelty_score -= 0.16
    novelty_score = max(0.08, min(0.95, novelty_score))

    direct_match = 0.45
    if seed_artists or seed_tracks:
        direct_match += 0.18
    if any(phrase in lowered for phrase in REFERENCE_PHRASES):
        direct_match += 0.08
    if re.search(r"^\s*[a-z0-9 .&'-]{2,40}\s*$", lowered) and (seed_artists or genres):
        direct_match += 0.06
    if "more like this" in lowered or "same vibe" in lowered:
        direct_match -= 0.06
    direct_match = max(0.12, min(0.85, direct_match))

    follow_up = any(phrase in lowered for phrase in FOLLOW_UP_PHRASES)
    if follow_up and session_context.get("current_song_name"):
        seed_tracks = _dedupe(seed_tracks + [str(session_context["current_song_name"])])
        current_artist = str(session_context.get("current_song_artists", "") or "").split(";")[0].strip()
        if current_artist and "more like this" in lowered:
            seed_artists = _dedupe(seed_artists + [current_artist])
        current_genre = str(session_context.get("current_song_genre", "") or "").strip()
        if current_genre and "same vibe" in lowered:
            genres = _dedupe(genres + [current_genre])

    if "more upbeat" in lowered or "more energy" in lowered:
        moods = _dedupe(moods + ["upbeat", "energetic"])
    if "less intense" in lowered or "softer" in lowered:
        moods = _dedupe(moods + ["smooth", "calm"])
    if "grimier" in lowered or "grittier" in lowered:
        moods = _dedupe(moods + ["dark", "aggressive"])
    if "more underground" in lowered or "less mainstream" in lowered:
        novelty_score = min(0.95, novelty_score + 0.12)
    if "same vibe" in lowered and session_context.get("last_moods"):
        moods = _dedupe(moods + list(session_context["last_moods"]))

    activity_context = [label for phrase, label in ACTIVITY_TERMS.items() if phrase in lowered]
    production_tags = [label for phrase, label in PRODUCTION_TERMS.items() if phrase in lowered]
    vocal_tags = [label for phrase, label in VOCAL_TERMS.items() if phrase in lowered]
    production_tags.extend(label for phrase, label in INSTRUMENTATION_TERMS.items() if phrase in lowered)

    energy_values = [ENERGY_MAP[key] for key in ENERGY_MAP if key in lowered]
    valence_values = [VALENCE_MAP[key] for key in VALENCE_MAP if key in lowered]
    tempo_values = [TEMPO_MAP[key] for key in TEMPO_MAP if key in lowered]

    energy_target = _weighted_mean(energy_values)
    valence_target = _weighted_mean(valence_values)
    tempo_target = _weighted_mean(tempo_values)

    if spec.constraints.get("energy"):
        lo, hi = spec.constraints["energy"]
        energy_target = float(lo + hi) / 2.0
    if spec.constraints.get("valence"):
        lo, hi = spec.constraints["valence"]
        valence_target = float(lo + hi) / 2.0
    if spec.constraints.get("tempo_bpm"):
        lo, hi = spec.constraints["tempo_bpm"]
        tempo_target = float(lo + hi) / 2.0

    era = None
    for pattern, label in ERA_PATTERNS:
        if re.search(pattern, lowered):
            era = label
            break

    requested_single_artist = bool(
        seed_artists
        and (
            "only" in lowered
            or "just" in lowered
            or "all kendrick" in lowered
            or "artist radio" in lowered
        )
    )

    excluded_artists: list[str] = []
    for phrase in ("no ", "not ", "without "):
        if phrase in lowered:
            tail = lowered.split(phrase, 1)[1]
            candidate = tail.split(",")[0].split(" but ")[0].split(" and ")[0].strip()
            if 2 <= len(candidate) <= 40:
                excluded_artists.append(candidate)

    similarity_mode = "direct" if direct_match >= 0.62 else "blend"
    if any(phrase in lowered for phrase in SIMILARITY_PHRASES):
        similarity_mode = "similar"

    semantic_terms = _dedupe(
        seed_artists
        + seed_tracks
        + genres
        + moods
        + activity_context
        + production_tags
        + vocal_tags
        + ([era] if era else [])
    )[:12]

    familiarity_target = max(0.05, min(0.92, 1.0 - novelty_score))
    exploration_weight = novelty_score

    return PromptIntent(
        seed_artists=seed_artists[:4],
        seed_tracks=seed_tracks[:3],
        excluded_artists=_dedupe(excluded_artists)[:3],
        genres=_dedupe(genres)[:6],
        moods=_dedupe(moods)[:8],
        energy_target=energy_target,
        valence_target=valence_target,
        tempo_target=tempo_target,
        novelty_target=novelty_score,
        familiarity_target=familiarity_target,
        activity_context=_dedupe(activity_context)[:3],
        era=era,
        production_tags=_dedupe(production_tags)[:4],
        vocal_tags=_dedupe(vocal_tags)[:4],
        direct_match_weight=direct_match,
        exploration_weight=exploration_weight,
        similarity_mode=similarity_mode,
        requested_single_artist=requested_single_artist,
        follow_up=follow_up,
        semantic_terms=semantic_terms,
    )
