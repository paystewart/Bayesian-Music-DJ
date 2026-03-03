from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .embedder import SemanticEmbedder

KNOWN_GENRES = [
    "acoustic",
    "afrobeats",
    "alt pop",
    "alternative",
    "ambient",
    "americana",
    "bass",
    "blues",
    "bollywood",
    "chillwave",
    "classical",
    "club",
    "country",
    "dance",
    "dance pop",
    "deep house",
    "disco",
    "drill",
    "drum and bass",
    "dubstep",
    "edm",
    "electronic",
    "emo",
    "folk",
    "french pop",
    "funk",
    "garage",
    "gospel",
    "grime",
    "grunge",
    "hard rock",
    "heavy metal",
    "hip hop",
    "house",
    "hyperpop",
    "indie",
    "indie folk",
    "indie pop",
    "j-pop",
    "jazz",
    "k-pop",
    "latin",
    "latin pop",
    "lofi",
    "melodic techno",
    "metal",
    "minimal techno",
    "neo soul",
    "new wave",
    "orchestral",
    "phonk",
    "piano",
    "pop",
    "pop punk",
    "post punk",
    "progressive house",
    "psytrance",
    "punk",
    "r&b",
    "rap",
    "reggae",
    "reggaeton",
    "rock",
    "singer songwriter",
    "soul",
    "synthwave",
    "tech house",
    "techno",
    "trance",
    "trap",
    "trip hop",
    "uk garage",
    "vaporwave",
]

KNOWN_MOODS = [
    "acoustic",
    "aggressive",
    "calm",
    "chill",
    "confident",
    "dark",
    "dreamy",
    "energetic",
    "epic",
    "focus",
    "happy",
    "hype",
    "melancholic",
    "moody",
    "party",
    "peaceful",
    "romantic",
    "sad",
    "smooth",
    "upbeat",
    "uplifting",
    "workout",
]

FEATURE_HINTS: dict[str, dict[str, tuple[float | int, float | int]]] = {
    "chill": {
        "energy": (0.2, 0.55),
        "acousticness": (0.35, 1.0),
    },
    "calm": {
        "energy": (0.1, 0.45),
        "tempo_bpm": (70, 110),
    },
    "focus": {
        "energy": (0.25, 0.6),
        "acousticness": (0.2, 1.0),
    },
    "party": {
        "energy": (0.7, 1.0),
        "danceability": (0.65, 1.0),
        "tempo_bpm": (110, 150),
    },
    "workout": {
        "energy": (0.75, 1.0),
        "danceability": (0.55, 1.0),
        "tempo_bpm": (120, 180),
    },
    "energetic": {
        "energy": (0.72, 1.0),
        "tempo_bpm": (115, 175),
    },
    "upbeat": {
        "energy": (0.65, 1.0),
        "valence": (0.6, 1.0),
    },
    "happy": {
        "valence": (0.62, 1.0),
    },
    "sad": {
        "valence": (0.0, 0.4),
        "energy": (0.1, 0.6),
    },
    "melancholic": {
        "valence": (0.0, 0.45),
    },
    "romantic": {
        "valence": (0.35, 0.75),
    },
    "aggressive": {
        "energy": (0.75, 1.0),
        "valence": (0.2, 0.75),
    },
    "confident": {
        "energy": (0.6, 0.9),
        "valence": (0.5, 0.85),
    },
    "dark": {
        "valence": (0.0, 0.35),
        "energy": (0.3, 0.75),
    },
    "dreamy": {
        "energy": (0.15, 0.5),
        "acousticness": (0.3, 1.0),
        "instrumentalness": (0.1, 1.0),
    },
    "epic": {
        "energy": (0.65, 1.0),
    },
    "hype": {
        "energy": (0.78, 1.0),
        "danceability": (0.6, 1.0),
    },
    "moody": {
        "valence": (0.1, 0.5),
        "energy": (0.25, 0.65),
    },
    "peaceful": {
        "energy": (0.05, 0.35),
        "acousticness": (0.4, 1.0),
    },
    "smooth": {
        "energy": (0.2, 0.55),
        "acousticness": (0.25, 0.8),
    },
    "uplifting": {
        "valence": (0.6, 1.0),
        "energy": (0.55, 0.9),
    },
    "not too upbeat": {
        "energy": (0.2, 0.62),
        "valence": (0.15, 0.65),
    },
    "not upbeat": {
        "energy": (0.1, 0.55),
        "valence": (0.05, 0.5),
    },
    "not too energetic": {
        "energy": (0.2, 0.6),
    },
    "not too high energy": {
        "energy": (0.2, 0.65),
    },
    "instrumental": {
        "instrumentalness": (0.7, 1.0),
        "speechiness": (0.0, 0.1),
    },
    "no vocals": {
        "instrumentalness": (0.7, 1.0),
        "speechiness": (0.0, 0.1),
    },
    "vocal": {
        "instrumentalness": (0.0, 0.3),
    },
}

GENRE_ALIASES: dict[str, str] = {
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "rnb": "r&b",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "drum & bass": "drum and bass",
    "drum n bass": "drum and bass",
    "dnb": "drum and bass",
    "d&b": "drum and bass",
    "electro": "electronic",
    "lo-fi": "lofi",
    "lo fi": "lofi",
    "synth wave": "synthwave",
    "synth-wave": "synthwave",
    "deep-house": "deep house",
    "pop-punk": "pop punk",
    "alt-pop": "alt pop",
    "indie-pop": "indie pop",
    "indie-folk": "indie folk",
    "singer-songwriter": "singer songwriter",
    "neosoul": "neo soul",
    "neo-soul": "neo soul",
    "kpop": "k-pop",
    "jpop": "j-pop",
    "post-punk": "post punk",
    "hard-rock": "hard rock",
    "trip-hop": "trip hop",
    "tech-house": "tech house",
    "uk-garage": "uk garage",
    "progressive-house": "progressive house",
    "melodic-techno": "melodic techno",
    "minimal-techno": "minimal techno",
    "dance-pop": "dance pop",
    "latin-pop": "latin pop",
    "french-pop": "french pop",
    "heavy-metal": "heavy metal",
}

FEATURE_ALIASES: dict[str, str] = {
    "energy": "energy",
    "valence": "valence",
    "danceability": "danceability",
    "danceable": "danceability",
    "acousticness": "acousticness",
    "acoustic": "acousticness",
    "instrumentalness": "instrumentalness",
    "instrumental": "instrumentalness",
    "speechiness": "speechiness",
    "liveness": "liveness",
    "loudness": "loudness",
}

_POPULARITY_HIGH_PHRASES = [
    "popular", "mainstream", "top hits", "top charts", "chart-topping",
    "chart topping", "top 40", "well known", "well-known", "hit songs",
]
_POPULARITY_LOW_PHRASES = [
    "underground", "obscure", "hidden gems", "hidden gem", "deep cuts",
    "deep cut", "lesser known", "lesser-known", "underrated", "niche",
]


@dataclass
class QuerySpec:
    seed_track: str | None = None
    seed_artists: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    constraints: dict[str, tuple[float | int, float | int]] = field(default_factory=dict)
    year_range: tuple[int, int] | None = None
    playlist_length: int = 30
    spotify_search_queries: list[str] = field(default_factory=list)

    def to_spotify_params(self) -> dict[str, Any]:
        """Flat dict of min_/max_/target_ keys for the Spotify Recommendations API."""
        params: dict[str, Any] = {}
        for feature, (lo, hi) in self.constraints.items():
            api_key = "tempo" if feature == "tempo_bpm" else feature
            params[f"min_{api_key}"] = lo
            params[f"max_{api_key}"] = hi
            mid = (lo + hi) / 2
            if api_key == "tempo":
                params[f"target_{api_key}"] = int(round(mid))
            elif isinstance(lo, int) and isinstance(hi, int):
                params[f"target_{api_key}"] = int(round(mid))
            else:
                params[f"target_{api_key}"] = round(mid, 3)
        if self.playlist_length:
            params["limit"] = min(self.playlist_length, 100)
        return params

    def to_spotify_search_queries(self, max_queries: int = 6) -> list[str]:
        queries: list[str] = []

        year_fragment = ""
        if self.year_range:
            year_fragment = f" year:{self.year_range[0]}-{self.year_range[1]}"

        if self.seed_track:
            q = f"track:{self.seed_track}"
            if self.seed_artists:
                q += f" artist:{self.seed_artists[0]}"
            queries.append(q)

        for artist in self.seed_artists[:3]:
            queries.append(f"artist:{artist}{year_fragment}".strip())

        for genre in self.genres[:3]:
            queries.append(f"genre:{genre}{year_fragment}".strip())

        if self.moods:
            if self.genres:
                for mood in self.moods[:2]:
                    if mood == self.genres[0]:
                        continue
                    queries.append(f"{mood} {self.genres[0]}{year_fragment}".strip())
            else:
                for mood in self.moods[:2]:
                    queries.append(f"{mood}{year_fragment}".strip())

        if self.year_range and not self.genres and not self.seed_artists:
            queries.append(f"year:{self.year_range[0]}-{self.year_range[1]}")

        if not queries:
            queries.append("genre:pop")

        deduped: list[str] = []
        seen: set[str] = set()
        for q in queries:
            if q in seen:
                continue
            seen.add(q)
            deduped.append(q)
            if len(deduped) >= max_queries:
                break

        if len(deduped) == 1:
            fallback = f"{deduped[0]} playlist"
            if fallback not in seen:
                deduped.append(fallback)

        return deduped

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["spotify_search_queries"]:
            data["spotify_search_queries"] = self.to_spotify_search_queries()
        data["spotify_params"] = self.to_spotify_params()
        return data


class MusicQueryParser:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = ".cache/music_query_parser",
        default_playlist_length: int = 30,
    ) -> None:
        self.default_playlist_length = default_playlist_length
        self.embedder = SemanticEmbedder(model_name=model_name, cache_dir=cache_dir, allow_fallback=True)
        self.genre_index = self.embedder.build_index("genres", KNOWN_GENRES)
        self.mood_index = self.embedder.build_index("moods", KNOWN_MOODS)

    def parse(self, prompt: str) -> QuerySpec:
        text = " ".join(prompt.strip().split())
        lowered = text.lower()

        seed_track = self._extract_seed_track(text)
        seed_artists = self._extract_seed_artists(text)
        year_range = self._extract_year_range(lowered)
        playlist_length = self._extract_playlist_length(lowered) or self.default_playlist_length

        genres = self._extract_labels(lowered, self.genre_index, KNOWN_GENRES, threshold=0.35, top_k=3)
        moods = self._extract_labels(lowered, self.mood_index, KNOWN_MOODS, threshold=0.30, top_k=4)
        moods = self._drop_negated_labels(lowered, moods)

        constraints: dict[str, tuple[float | int, float | int]] = {}
        self._extract_explicit_constraints(lowered, constraints)
        self._apply_feature_hints(lowered, genres + moods, constraints)

        spec = QuerySpec(
            seed_track=seed_track,
            seed_artists=seed_artists,
            genres=genres,
            moods=moods,
            constraints=constraints,
            year_range=year_range,
            playlist_length=playlist_length,
        )
        spec.spotify_search_queries = spec.to_spotify_search_queries()
        return spec

    def _extract_labels(
        self,
        lowered_prompt: str,
        index,
        labels: list[str],
        threshold: float,
        top_k: int,
    ) -> list[str]:
        explicit: list[str] = []
        for alias, canonical in GENRE_ALIASES.items():
            if canonical in labels and re.search(rf"\b{re.escape(alias)}\b", lowered_prompt):
                explicit.append(canonical)
        for label in labels:
            if re.search(rf"\b{re.escape(label)}\b", lowered_prompt):
                explicit.append(label)

        semantic_matches = self.embedder.similarity_search(
            lowered_prompt,
            index=index,
            top_k=top_k,
            min_score=threshold,
        )
        for label, _ in semantic_matches:
            explicit.append(label)

        seen: set[str] = set()
        deduped: list[str] = []
        for tag in explicit:
            if tag in seen:
                continue
            seen.add(tag)
            deduped.append(tag)
        return deduped[:top_k]

    def _extract_seed_track(self, text: str) -> str | None:
        quoted = re.findall(r'["\u201c\u2018\']([^"\u201d\u2019\']{1,120})["\u201d\u2019\']', text)
        if quoted:
            return quoted[0].strip()

        start_with = re.search(
            r"\b(?:start with|open with|begin with)\s+"
            r"([a-zA-Z0-9&.'\- ]{2,80}?)"
            r"(?:\s+(?:then|and then|followed by|and|but|,)|$)",
            text,
            flags=re.IGNORECASE,
        )
        if start_with:
            return self._clean_entity(start_with.group(1))
        return None

    def _extract_seed_artists(self, text: str) -> list[str]:
        patterns = [
            r"\b(?:like|similar to|in the style of|by)\s+([a-zA-Z0-9&.'\- ]{2,120})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                raw = match.group(1)
                parts = re.split(r"\s*(?:,\s*|\s+and\s+|\s*&\s*)", raw)
                artists: list[str] = []
                for part in parts:
                    cleaned = self._clean_entity(part)
                    if cleaned and len(cleaned) >= 2:
                        artists.append(cleaned)
                    if len(artists) >= 5:
                        break
                if artists:
                    return artists
        return []

    def _clean_entity(self, raw: str) -> str:
        stop_phrases = [
            "then similar", "then ", "followed by", "not too", "with a",
            "playlist", "songs", "tracks", "from the",
        ]
        result = raw
        for phrase in stop_phrases:
            idx = result.lower().find(phrase)
            if idx > 0:
                result = result[:idx]
        cleaned = re.split(r"[.!;]|(?:\bbut\b)", result, maxsplit=1)[0]
        return cleaned.strip(" ',\"")

    def _extract_year_range(self, lowered: str) -> tuple[int, int] | None:
        m = re.search(r"\b(19\d{2}|20\d{2})\s*(?:-|to|\u2013|\u2014)\s*(19\d{2}|20\d{2})\b", lowered)
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            return (min(start, end), max(start, end))

        decade = re.search(r"\b(?:the\s+)?['\u2019]?(\d{2})s\b", lowered)
        if decade:
            d = int(decade.group(1))
            century = 1900 if d >= 20 else 2000
            base = century + d
            return (base, base + 9)

        single = re.search(
            r"\b(?:from|in|released(?: in)?|circa)\s+(19\d{2}|20\d{2})\b",
            lowered,
        )
        if single:
            year = int(single.group(1))
            return (year, year)

        return None

    def _extract_playlist_length(self, lowered: str) -> int | None:
        m = re.search(r"\b(\d{1,3})\s*(?:songs|song|tracks|track)\b", lowered)
        if not m:
            m = re.search(r"\bplaylist(?: of)?\s*(\d{1,3})\b", lowered)
        if not m:
            return None
        length = int(m.group(1))
        return max(1, min(length, 200))

    def _extract_explicit_constraints(
        self,
        lowered: str,
        constraints: dict[str, tuple[float | int, float | int]],
    ) -> None:
        bpm_range = re.search(r"\b(\d{2,3})\s*(?:-|to|\u2013|\u2014)\s*(\d{2,3})\s*bpm\b", lowered)
        if bpm_range:
            lo, hi = int(bpm_range.group(1)), int(bpm_range.group(2))
            self._merge_range(constraints, "tempo_bpm", (min(lo, hi), max(lo, hi)))
        else:
            bpm_single = re.search(r"\b(\d{2,3})\s*bpm\b", lowered)
            if bpm_single:
                bpm = int(bpm_single.group(1))
                self._merge_range(constraints, "tempo_bpm", (max(40, bpm - 8), min(220, bpm + 8)))

        all_features = "|".join(FEATURE_ALIASES.keys())
        numeric_feature = re.finditer(
            rf"\b({all_features})\s*(?:between|from)?\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*(?:-|to|and)\s*(0(?:\.\d+)?|1(?:\.0+)?)",
            lowered,
        )
        for match in numeric_feature:
            feature = FEATURE_ALIASES[match.group(1)]
            lo = float(match.group(2))
            hi = float(match.group(3))
            self._merge_range(constraints, feature, (min(lo, hi), max(lo, hi)))

        level_feature = re.finditer(
            rf"\b(very high|high|medium|low)\s+({all_features})\b",
            lowered,
        )
        level_map = {
            "very high": (0.82, 1.0),
            "high": (0.68, 1.0),
            "medium": (0.35, 0.7),
            "low": (0.0, 0.4),
        }
        for match in level_feature:
            level = match.group(1)
            feature = FEATURE_ALIASES[match.group(2)]
            self._merge_range(constraints, feature, level_map[level])

        not_too_feature = re.finditer(
            r"\bnot too\s+(energy|energetic|upbeat|danceable|danceability)\b",
            lowered,
        )
        for match in not_too_feature:
            token = match.group(1)
            if token in ("energy", "energetic", "upbeat"):
                self._merge_range(constraints, "energy", (0.15, 0.62))
            else:
                self._merge_range(constraints, "danceability", (0.2, 0.68))

        if re.search(r"\b(?:instrumental|no vocals|no singing)\b", lowered):
            if not self._is_negated_phrase(lowered, "instrumental"):
                self._merge_range(constraints, "instrumentalness", (0.7, 1.0))
                self._merge_range(constraints, "speechiness", (0.0, 0.1))

        if re.search(r"\b(?:with vocals|with singing|singer|singers)\b", lowered):
            self._merge_range(constraints, "instrumentalness", (0.0, 0.3))

        if re.search(r"\b(?:spoken word)\b", lowered):
            self._merge_range(constraints, "speechiness", (0.6, 1.0))

        self._extract_popularity(lowered, constraints)

    def _extract_popularity(
        self,
        lowered: str,
        constraints: dict[str, tuple[float | int, float | int]],
    ) -> None:
        for phrase in _POPULARITY_HIGH_PHRASES:
            if phrase in lowered:
                self._merge_range(constraints, "popularity", (60, 100))
                return
        for phrase in _POPULARITY_LOW_PHRASES:
            if phrase in lowered:
                self._merge_range(constraints, "popularity", (0, 35))
                return

    def _apply_feature_hints(
        self,
        lowered: str,
        tags: list[str],
        constraints: dict[str, tuple[float | int, float | int]],
    ) -> None:
        for phrase in sorted(FEATURE_HINTS, key=len, reverse=True):
            if phrase in lowered:
                if not phrase.startswith("not ") and self._is_negated_phrase(lowered, phrase):
                    continue
                for feature, new_range in FEATURE_HINTS[phrase].items():
                    self._merge_range(constraints, feature, new_range)

        for tag in tags:
            if tag in FEATURE_HINTS:
                if self._is_negated_phrase(lowered, tag):
                    continue
                for feature, new_range in FEATURE_HINTS[tag].items():
                    self._merge_range(constraints, feature, new_range)

    def _drop_negated_labels(self, lowered: str, labels: list[str]) -> list[str]:
        return [label for label in labels if not self._is_negated_phrase(lowered, label)]

    def _is_negated_phrase(self, lowered: str, phrase: str) -> bool:
        return bool(re.search(rf"\bnot(?:\s+too)?\s+{re.escape(phrase)}\b", lowered))

    def _merge_range(
        self,
        constraints: dict[str, tuple[float | int, float | int]],
        key: str,
        new_range: tuple[float | int, float | int],
    ) -> None:
        if key not in constraints:
            constraints[key] = new_range
            return
        old_lo, old_hi = constraints[key]
        new_lo, new_hi = new_range
        lo = max(float(old_lo), float(new_lo))
        hi = min(float(old_hi), float(new_hi))
        if lo <= hi:
            if key in ("tempo_bpm", "popularity"):
                constraints[key] = (int(round(lo)), int(round(hi)))
            else:
                constraints[key] = (round(lo, 3), round(hi, 3))
        else:
            constraints[key] = new_range
