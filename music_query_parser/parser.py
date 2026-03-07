from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from music_query_parser.embedder import SemanticEmbedder

KNOWN_GENRES = [
    "acoustic",
    "adult contemporary",
    "afrobeats",
    "afro house",
    "amapiano",
    "alt r&b",
    "alt pop",
    "alternative",
    "ambient",
    "americana",
    "anthemic pop",
    "art pop",
    "bass",
    "baile funk",
    "baltimore club",
    "bedroom pop",
    "big room",
    "blues",
    "bollywood",
    "boom bap",
    "bossa nova",
    "brazilian funk",
    "chill house",
    "chillwave",
    "chicago drill",
    "christian hip hop",
    "christian pop",
    "classical",
    "club",
    "cloud rap",
    "contemporary r&b",
    "country",
    "country pop",
    "dancehall",
    "dance",
    "dance pop",
    "dance rap",
    "dark pop",
    "deep house",
    "detroit rap",
    "disco",
    "drill",
    "dream pop",
    "drum and bass",
    "dubstep",
    "edm",
    "east coast hip hop",
    "electronic",
    "electro house",
    "electro pop",
    "emo",
    "emo rap",
    "experimental",
    "festival edm",
    "folk",
    "funk rock",
    "future bass",
    "future house",
    "gangsta rap",
    "glitchcore",
    "french pop",
    "funk",
    "garage",
    "glo",
    "gospel",
    "gospel rap",
    "grime",
    "grunge",
    "hard rock",
    "hard techno",
    "hardstyle",
    "heavy metal",
    "hip hop",
    "house",
    "industrial",
    "hyperpop",
    "indie",
    "indie folk",
    "indie pop",
    "indie rock",
    "jazz rap",
    "j-pop",
    "jersey club",
    "jazz",
    "k-pop",
    "latin",
    "latin pop",
    "liquid drum and bass",
    "lo fi hip hop",
    "lofi",
    "lounge",
    "lounge house",
    "melodic techno",
    "melodic house",
    "metal",
    "miami bass",
    "minimal techno",
    "modern rock",
    "neo soul",
    "new jazz",
    "new wave",
    "orchestral",
    "phonk",
    "piano",
    "pop",
    "pop country",
    "pop punk",
    "pop rap",
    "post punk",
    "pluggnb",
    "progressive house",
    "progressive rock",
    "progressive trance",
    "psychedelic rock",
    "psytrance",
    "punk",
    "rage",
    "r&b",
    "rap",
    "reggae",
    "reggaeton",
    "road trip rock",
    "rock",
    "sad rap",
    "shoegaze",
    "singer songwriter",
    "soft rock",
    "soul",
    "southern hip hop",
    "southern rock",
    "stutter house",
    "synthwave",
    "techno rap",
    "tech house",
    "techno",
    "trance",
    "trap",
    "trap soul",
    "trip hop",
    "uk rap",
    "uk garage",
    "uk drill",
    "underground rap",
    "vaporwave",
    "west coast hip hop",
    "west coast rap",
]

KNOWN_MOODS = [
    "acoustic",
    "aggressive",
    "after hours",
    "afternoon",
    "afterparty",
    "airy",
    "beach",
    "bedroom",
    "bold",
    "calm",
    "carefree",
    "chill",
    "cinematic",
    "clubby",
    "confident",
    "commute",
    "cozy",
    "dark",
    "date night",
    "daytime",
    "dreamy",
    "driving",
    "early morning",
    "energetic",
    "epic",
    "evening",
    "feel good",
    "floaty",
    "focus",
    "friday",
    "friday night",
    "girls night",
    "gloomy",
    "golden hour",
    "happy",
    "hard",
    "hazy",
    "heartbroken",
    "hype",
    "intimate",
    "late afternoon",
    "late night",
    "lazy sunday",
    "light",
    "lit",
    "lonely",
    "loud",
    "melancholic",
    "mellow",
    "midnight",
    "moody",
    "monday morning",
    "morning",
    "night drive",
    "nostalgic",
    "open road",
    "party starter",
    "poolside",
    "preparty",
    "pregame",
    "punchy",
    "rainy",
    "rainy afternoon",
    "rainy evening",
    "rainy night",
    "raw",
    "rebellious",
    "reflective",
    "relaxed",
    "road trip",
    "rooftop",
    "party",
    "peaceful",
    "powerful",
    "seductive",
    "sexy",
    "romantic",
    "sad",
    "saturday",
    "saturday night",
    "slow burn",
    "soft",
    "sparkly",
    "stormy",
    "sunday morning",
    "sunday afternoon",
    "smooth",
    "sunrise",
    "sunset",
    "summer",
    "summer night",
    "sweaty",
    "thursday",
    "thursday night",
    "tuesday",
    "tuesday afternoon",
    "turn up",
    "upbeat",
    "uplifting",
    "vulnerable",
    "warm",
    "weekday",
    "weekend",
    "wednesday",
    "wednesday night",
    "wild",
    "workout",
]

MOOD_ALIASES: dict[str, list[str]] = {
    "monday": ["focus", "commute"],
    "tuesday": ["focus", "daytime"],
    "wednesday": ["focus", "moody"],
    "thursday": ["party", "upbeat"],
    "hard": ["aggressive", "hype"],
    "hardcore": ["aggressive", "hype"],
    "heavy": ["aggressive", "powerful"],
    "friday": ["party", "upbeat"],
    "friday night": ["party", "hype"],
    "saturday": ["party", "confident"],
    "saturday night": ["party", "hype"],
    "sunday": ["peaceful", "warm"],
    "sunday morning": ["peaceful", "acoustic", "warm"],
    "monday morning": ["focus", "commute"],
    "tuesday afternoon": ["daytime", "focus"],
    "wednesday night": ["moody", "smooth"],
    "thursday night": ["party", "hype"],
    "afternoon": ["daytime", "warm"],
    "late afternoon": ["warm", "dreamy"],
    "evening": ["moody", "smooth"],
    "morning": ["uplifting", "peaceful"],
    "early morning": ["peaceful", "light", "focus"],
    "sunrise": ["peaceful", "uplifting"],
    "sunset": ["warm", "dreamy"],
    "afterparty": ["party", "dark"],
    "preparty": ["party", "upbeat"],
    "after hours": ["late night", "moody", "smooth"],
    "weekend": ["party", "hype"],
    "weekday": ["focus", "commute"],
    "going out": ["party", "hype", "upbeat"],
    "pregame": ["party", "hype", "upbeat"],
    "turn up": ["party", "hype", "energetic"],
    "turnt": ["party", "hype", "energetic"],
    "lit": ["party", "hype", "wild"],
    "wild": ["party", "energetic", "aggressive"],
    "club": ["party", "clubby", "energetic"],
    "rooftop": ["party", "dreamy", "upbeat"],
    "poolside": ["summer", "upbeat", "warm"],
    "beach": ["summer", "uplifting", "warm"],
    "night drive": ["driving", "late night", "moody"],
    "open road": ["driving", "energetic", "uplifting"],
    "road trip": ["driving", "upbeat", "energetic"],
    "commute": ["focus", "daytime"],
    "girls night": ["party", "confident", "upbeat"],
    "lowkey": ["chill", "moody"],
    "low key": ["chill", "moody"],
    "easygoing": ["relaxed", "warm"],
    "carefree": ["feel good", "uplifting"],
    "rainy": ["moody", "melancholic", "chill"],
    "rainy afternoon": ["melancholic", "relaxed", "mellow"],
    "rainy evening": ["moody", "smooth", "chill"],
    "rainy night": ["dark", "smooth", "moody"],
    "stormy": ["dark", "aggressive", "cinematic"],
    "late night": ["moody", "smooth"],
    "bedroom": ["intimate", "smooth", "dreamy"],
    "cozy": ["warm", "peaceful", "chill"],
    "date night": ["romantic", "seductive", "smooth"],
    "slow burn": ["seductive", "intimate", "moody"],
    "raw": ["aggressive", "dark"],
    "seductive": ["sexy", "smooth", "romantic"],
    "floaty": ["dreamy", "airy", "chill"],
    "hazy": ["dreamy", "nostalgic", "smooth"],
    "gloomy": ["melancholic", "dark", "moody"],
    "heartbroken": ["sad", "vulnerable", "melancholic"],
    "feel good": ["happy", "uplifting", "warm"],
    "summer night": ["warm", "party", "dreamy"],
    "soft": ["peaceful", "chill", "intimate"],
    "punchy": ["confident", "energetic", "upbeat"],
    "rebellious": ["aggressive", "confident", "wild"],
    "gritty": ["raw", "aggressive", "dark"],
    "cinematic": ["epic", "dreamy", "powerful"],
    "ambient": ["calm", "dreamy", "focus"],
    "headbanger": ["aggressive", "wild", "powerful"],
    "moshpit": ["aggressive", "wild", "energetic"],
    "midnight": ["late night", "dark", "smooth"],
}

DOMINANT_SUBGENRES: dict[str, set[str]] = {
    "drill": {"hip hop", "rap", "trap"},
    "phonk": {"hip hop", "rap", "trap"},
    "grime": {"hip hop", "rap"},
    "trap": {"hip hop", "rap"},
    "gangsta rap": {"hip hop", "rap"},
    "southern hip hop": {"hip hop", "rap"},
    "east coast hip hop": {"hip hop", "rap"},
    "west coast hip hop": {"hip hop", "rap"},
    "west coast rap": {"hip hop", "rap"},
    "detroit rap": {"hip hop", "rap"},
    "pop rap": {"hip hop", "rap", "pop"},
    "jazz rap": {"hip hop", "rap", "jazz"},
    "trap soul": {"r&b", "soul", "trap"},
    "contemporary r&b": {"r&b"},
    "alt r&b": {"r&b"},
    "hard rock": {"rock"},
    "heavy metal": {"metal", "rock"},
    "soft rock": {"rock"},
    "indie rock": {"indie", "rock"},
    "shoegaze": {"rock", "alternative"},
    "dream pop": {"pop", "alternative"},
    "deep house": {"house", "dance", "electronic"},
    "progressive house": {"house", "dance", "electronic"},
    "tech house": {"house", "dance", "electronic"},
    "chill house": {"house", "dance", "electronic"},
    "melodic house": {"house", "dance", "electronic"},
    "melodic techno": {"techno", "electronic"},
    "minimal techno": {"techno", "electronic"},
    "indie pop": {"indie", "pop"},
    "indie folk": {"indie", "folk"},
    "dance pop": {"dance", "pop"},
    "country pop": {"country", "pop"},
    "electro pop": {"electronic", "pop"},
    "electro house": {"house", "electronic"},
    "liquid drum and bass": {"drum and bass"},
}

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
    "hard": {
        "energy": (0.78, 1.0),
        "danceability": (0.45, 0.9),
        "acousticness": (0.0, 0.22),
        "instrumentalness": (0.0, 0.18),
        "tempo_bpm": (118, 160),
    },
    "friday night": {
        "energy": (0.72, 1.0),
        "danceability": (0.58, 1.0),
        "tempo_bpm": (110, 155),
    },
    "saturday night": {
        "energy": (0.72, 1.0),
        "danceability": (0.6, 1.0),
        "tempo_bpm": (112, 158),
    },
    "thursday night": {
        "energy": (0.64, 1.0),
        "danceability": (0.56, 1.0),
        "tempo_bpm": (108, 150),
    },
    "pregame": {
        "energy": (0.72, 1.0),
        "danceability": (0.58, 1.0),
        "valence": (0.48, 0.95),
        "tempo_bpm": (108, 156),
    },
    "preparty": {
        "energy": (0.68, 1.0),
        "danceability": (0.56, 1.0),
        "tempo_bpm": (106, 152),
    },
    "turn up": {
        "energy": (0.82, 1.0),
        "danceability": (0.6, 1.0),
        "tempo_bpm": (118, 170),
    },
    "lit": {
        "energy": (0.82, 1.0),
        "danceability": (0.58, 1.0),
        "tempo_bpm": (118, 170),
    },
    "drill": {
        "energy": (0.72, 1.0),
        "danceability": (0.45, 0.82),
        "valence": (0.12, 0.58),
        "acousticness": (0.0, 0.18),
        "speechiness": (0.18, 0.6),
        "instrumentalness": (0.0, 0.12),
        "tempo_bpm": (128, 170),
    },
    "rap": {
        "energy": (0.55, 0.95),
        "danceability": (0.55, 0.92),
        "acousticness": (0.0, 0.28),
        "speechiness": (0.12, 0.55),
        "instrumentalness": (0.0, 0.12),
    },
    "hip hop": {
        "energy": (0.5, 0.92),
        "danceability": (0.55, 0.9),
        "acousticness": (0.0, 0.25),
        "speechiness": (0.1, 0.48),
        "instrumentalness": (0.0, 0.12),
    },
    "trap": {
        "energy": (0.64, 1.0),
        "danceability": (0.55, 0.9),
        "acousticness": (0.0, 0.2),
        "speechiness": (0.12, 0.5),
        "tempo_bpm": (120, 165),
    },
    "gangsta rap": {
        "energy": (0.62, 1.0),
        "danceability": (0.48, 0.85),
        "speechiness": (0.14, 0.58),
        "acousticness": (0.0, 0.22),
    },
    "rage": {
        "energy": (0.82, 1.0),
        "danceability": (0.48, 0.86),
        "acousticness": (0.0, 0.15),
        "tempo_bpm": (130, 175),
    },
    "hard rock": {
        "energy": (0.7, 1.0),
        "valence": (0.28, 0.78),
        "acousticness": (0.0, 0.22),
        "instrumentalness": (0.0, 0.28),
    },
    "indie rock": {
        "energy": (0.42, 0.82),
        "valence": (0.28, 0.74),
        "acousticness": (0.08, 0.52),
    },
    "dream pop": {
        "energy": (0.18, 0.58),
        "acousticness": (0.12, 0.7),
        "instrumentalness": (0.02, 0.42),
    },
    "shoegaze": {
        "energy": (0.34, 0.8),
        "valence": (0.1, 0.52),
        "instrumentalness": (0.02, 0.36),
    },
    "neo soul": {
        "energy": (0.22, 0.62),
        "danceability": (0.4, 0.78),
        "valence": (0.24, 0.7),
        "acousticness": (0.18, 0.7),
    },
    "alt r&b": {
        "energy": (0.24, 0.66),
        "danceability": (0.42, 0.82),
        "valence": (0.12, 0.62),
        "acousticness": (0.06, 0.54),
    },
    "synthwave": {
        "energy": (0.42, 0.86),
        "valence": (0.24, 0.72),
        "instrumentalness": (0.02, 0.62),
    },
    "house": {
        "energy": (0.6, 0.95),
        "danceability": (0.6, 0.95),
        "tempo_bpm": (118, 132),
    },
    "deep house": {
        "energy": (0.42, 0.8),
        "danceability": (0.62, 0.92),
        "tempo_bpm": (116, 126),
    },
    "melodic house": {
        "energy": (0.46, 0.84),
        "danceability": (0.52, 0.84),
        "tempo_bpm": (116, 128),
    },
    "techno": {
        "energy": (0.62, 1.0),
        "danceability": (0.42, 0.82),
        "instrumentalness": (0.18, 1.0),
        "tempo_bpm": (122, 148),
    },
    "afrobeats": {
        "energy": (0.44, 0.86),
        "danceability": (0.58, 0.92),
        "valence": (0.42, 0.88),
        "tempo_bpm": (96, 126),
    },
    "country": {
        "energy": (0.26, 0.72),
        "valence": (0.24, 0.8),
        "acousticness": (0.18, 0.82),
    },
    "country pop": {
        "energy": (0.42, 0.82),
        "valence": (0.44, 0.9),
        "acousticness": (0.08, 0.52),
    },
    "moody": {
        "valence": (0.1, 0.5),
        "energy": (0.25, 0.65),
    },
    "mellow": {
        "energy": (0.12, 0.46),
        "valence": (0.14, 0.62),
        "acousticness": (0.18, 0.74),
    },
    "peaceful": {
        "energy": (0.05, 0.35),
        "acousticness": (0.4, 1.0),
    },
    "relaxed": {
        "energy": (0.12, 0.48),
        "danceability": (0.22, 0.66),
        "acousticness": (0.16, 0.82),
    },
    "smooth": {
        "energy": (0.2, 0.55),
        "acousticness": (0.25, 0.8),
    },
    "warm": {
        "energy": (0.18, 0.62),
        "valence": (0.32, 0.78),
        "acousticness": (0.16, 0.72),
    },
    "rainy": {
        "energy": (0.12, 0.48),
        "valence": (0.08, 0.42),
        "acousticness": (0.22, 0.82),
    },
    "rainy afternoon": {
        "energy": (0.1, 0.44),
        "valence": (0.08, 0.46),
        "acousticness": (0.22, 0.82),
    },
    "lowkey": {
        "energy": (0.12, 0.46),
        "valence": (0.08, 0.48),
        "acousticness": (0.18, 0.72),
    },
    "late night": {
        "energy": (0.12, 0.58),
        "valence": (0.06, 0.5),
        "acousticness": (0.08, 0.64),
    },
    "morning": {
        "energy": (0.18, 0.62),
        "valence": (0.42, 0.86),
        "tempo_bpm": (82, 126),
    },
    "monday morning": {
        "energy": (0.28, 0.62),
        "valence": (0.32, 0.72),
        "tempo_bpm": (86, 122),
    },
    "sunday morning": {
        "energy": (0.08, 0.42),
        "valence": (0.32, 0.78),
        "acousticness": (0.28, 0.92),
    },
    "afternoon": {
        "energy": (0.28, 0.68),
        "valence": (0.34, 0.8),
    },
    "evening": {
        "energy": (0.18, 0.58),
        "valence": (0.16, 0.62),
    },
    "golden hour": {
        "energy": (0.22, 0.58),
        "valence": (0.38, 0.82),
        "acousticness": (0.14, 0.68),
    },
    "sunset": {
        "energy": (0.22, 0.62),
        "valence": (0.24, 0.68),
    },
    "sunrise": {
        "energy": (0.12, 0.54),
        "valence": (0.38, 0.82),
        "acousticness": (0.2, 0.78),
    },
    "driving": {
        "energy": (0.34, 0.82),
        "tempo_bpm": (90, 142),
    },
    "night drive": {
        "energy": (0.24, 0.68),
        "valence": (0.1, 0.54),
        "tempo_bpm": (88, 132),
    },
    "open road": {
        "energy": (0.42, 0.86),
        "valence": (0.32, 0.84),
        "tempo_bpm": (92, 138),
    },
    "gloomy": {
        "energy": (0.12, 0.52),
        "valence": (0.0, 0.34),
    },
    "stormy": {
        "energy": (0.36, 0.84),
        "valence": (0.0, 0.34),
    },
    "date night": {
        "energy": (0.18, 0.58),
        "valence": (0.26, 0.7),
        "danceability": (0.34, 0.76),
    },
    "seductive": {
        "energy": (0.16, 0.58),
        "valence": (0.18, 0.62),
        "danceability": (0.32, 0.78),
    },
    "soft": {
        "energy": (0.05, 0.36),
        "acousticness": (0.3, 1.0),
    },
    "floaty": {
        "energy": (0.08, 0.42),
        "instrumentalness": (0.06, 0.84),
        "acousticness": (0.18, 0.82),
    },
    "cinematic": {
        "energy": (0.3, 0.84),
        "instrumentalness": (0.08, 1.0),
    },
    "wild": {
        "energy": (0.82, 1.0),
        "danceability": (0.46, 0.9),
        "tempo_bpm": (118, 175),
    },
    "powerful": {
        "energy": (0.72, 1.0),
        "valence": (0.22, 0.82),
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
    "alt-r&b": "alt r&b",
    "alternative r&b": "alt r&b",
    "adult contemporary pop": "adult contemporary",
    "afrobeat": "afrobeats",
    "afro beats": "afrobeats",
    "afro-house": "afro house",
    "a mapiano": "amapiano",
    "art-pop": "art pop",
    "baile-funk": "baile funk",
    "bedroom-pop": "bedroom pop",
    "boom-bap": "boom bap",
    "brazil funk": "brazilian funk",
    "brazilian-funk": "brazilian funk",
    "chill-house": "chill house",
    "cloud-rap": "cloud rap",
    "contemporary rnb": "contemporary r&b",
    "country-pop": "country pop",
    "dance-rap": "dance rap",
    "dark-pop": "dark pop",
    "detroit-rap": "detroit rap",
    "dream-pop": "dream pop",
    "east coast rap": "east coast hip hop",
    "east-coast hip hop": "east coast hip hop",
    "east-coast rap": "east coast hip hop",
    "electro-pop": "electro pop",
    "electro-house": "electro house",
    "festival edm": "festival edm",
    "funk-rock": "funk rock",
    "gospel-rap": "gospel rap",
    "hard-techno": "hard techno",
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "indie-rock": "indie rock",
    "jersey drill": "jersey club",
    "liquid dnb": "liquid drum and bass",
    "liquid drum & bass": "liquid drum and bass",
    "liquid drum n bass": "liquid drum and bass",
    "lofi hip hop": "lo fi hip hop",
    "lounge-house": "lounge house",
    "melodic-house": "melodic house",
    "rnb": "r&b",
    "r and b": "r&b",
    "rhythm and blues": "r&b",
    "new-jazz": "new jazz",
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
    "plugg n b": "pluggnb",
    "pluggnb": "pluggnb",
    "pop-country": "pop country",
    "post-punk": "post punk",
    "progressive-rock": "progressive rock",
    "hard-rock": "hard rock",
    "rage rap": "rage",
    "trip-hop": "trip hop",
    "shoegaze rock": "shoegaze",
    "soft-rock": "soft rock",
    "southern-hip hop": "southern hip hop",
    "tech-house": "tech house",
    "trap-soul": "trap soul",
    "uk-rap": "uk rap",
    "uk-garage": "uk garage",
    "west coast hip-hop": "west coast hip hop",
    "west coast hip hop": "west coast hip hop",
    "west-coast rap": "west coast rap",
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

    def to_spotify_search_queries(self, max_queries: int = 8) -> list[str]:
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

        for genre in self.genres[:4]:
            queries.append(f"genre:{genre}{year_fragment}".strip())

        if self.moods:
            if self.genres:
                for mood in self.moods[:3]:
                    if mood == self.genres[0]:
                        continue
                    queries.append(f"{mood} {self.genres[0]}{year_fragment}".strip())
            else:
                for mood in self.moods[:3]:
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

        genres = self._extract_labels(lowered, self.genre_index, KNOWN_GENRES, threshold=0.35, top_k=5)
        moods = self._extract_labels(lowered, self.mood_index, KNOWN_MOODS, threshold=0.30, top_k=6)
        moods.extend(self._extract_mood_aliases(lowered))
        moods = self._drop_negated_labels(lowered, moods)
        genres = self._reduce_redundant_genres(genres)
        moods = self._dedupe_preserve_order(moods)[:6]

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

    def _extract_mood_aliases(self, lowered_prompt: str) -> list[str]:
        moods: list[str] = []
        for phrase, mapped in MOOD_ALIASES.items():
            if phrase in lowered_prompt:
                moods.extend(mapped)
        return moods

    def _reduce_redundant_genres(self, genres: list[str]) -> list[str]:
        reduced = list(genres)
        for dominant, redundant in DOMINANT_SUBGENRES.items():
            if dominant in reduced:
                reduced = [genre for genre in reduced if genre == dominant or genre not in redundant]
        return self._dedupe_preserve_order(reduced)

    def _dedupe_preserve_order(self, labels: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for label in labels:
            if label in seen:
                continue
            seen.add(label)
            deduped.append(label)
        return deduped

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
