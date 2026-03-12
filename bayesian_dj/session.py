from __future__ import annotations

from pathlib import Path

import numpy as np

from music_query_parser.parser import MusicQueryParser, QuerySpec

from .diagnostics import generate_all_diagnostics
from .model import FEATURE_INDEX, BayesianLogisticRegression
from .song_pool import AUDIO_FEATURES, SongInfo, SongPool

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "kaggle_dataset.csv"


class DJSession:
    """Interactive Bayesian DJ session.

    Parses a user prompt, builds a candidate song pool, initialises a
    Bayesian logistic-regression prior from the prompt constraints, then
    runs a play/skip feedback loop with Thompson-sampling selection.

    Pass a pre-built *pool* (Spotify-backed UI path) or *csv_path* (CLI).
    If *pool* is provided, *csv_path* is ignored.
    """

    def __init__(
        self,
        csv_path: str | Path | None = DEFAULT_CSV,
        pool: SongPool | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = ".cache/music_query_parser",
        playlist_length: int = 30,
        analyze: bool = False,
        output_dir: str | Path = "output",
        parser: MusicQueryParser | None = None,
    ) -> None:
        self.parser = parser or MusicQueryParser(
            model_name=model_name, cache_dir=cache_dir
        )
        if pool is not None:
            self.pool = pool
        elif csv_path is not None:
            self.pool = SongPool(csv_path)
        else:
            raise ValueError("Either pool or csv_path must be provided.")
        self.playlist_length = playlist_length
        self.analyze = analyze
        self.output_dir = Path(output_dir)

        self.model: BayesianLogisticRegression | None = None
        self.spec: QuerySpec | None = None
        self.prompt_intent = None
        self.playlist: list[SongInfo] = []
        self.actions: list[str] = []
        self._current_song: SongInfo | None = None
        self._current_x: np.ndarray | None = None
        self._current_feedback_action: str | None = None
        self.last_recommendation_score: float | None = None

    def start(self, prompt: str) -> QuerySpec:
        """Parse the prompt, filter the song pool, and initialise the prior."""
        self.spec = self.parser.parse(prompt)

        self.pool.filter_by_genres(self.spec.genres)

        constraints = dict(self.spec.constraints)
        self.model = BayesianLogisticRegression.from_constraints(constraints)

        self.model.snapshot()

        return self.spec

    def recommend_next(
        self,
        preferred_artists: list[str] | None = None,
        require_artist_match: bool = False,
    ) -> SongInfo | None:
        """Use Thompson sampling to pick the next song."""
        if self.model is None:
            raise RuntimeError("Call start() before recommending songs.")
        if self.pool.n_available == 0:
            return None

        feat_matrix = self.pool.get_feature_matrix()
        posterior_scores = self.model.predict_proba_posterior(feat_matrix)
        thompson_scores = self.model.thompson_sample_scores(feat_matrix)
        popularity_scores = self.pool.get_popularity_scores()
        discovery_scores = self.pool.get_external_bias_scores()
        prior_scores = self._prior_alignment_scores(feat_matrix)
        coherence_scores = self._coherence_scores(feat_matrix)
        scores = (
            0.20 * thompson_scores
            + 0.20 * posterior_scores
            + 0.08 * prior_scores
            + 0.12 * popularity_scores
            + 0.30 * discovery_scores
            + 0.10 * coherence_scores
        )

        recent_artists: set[str] = set()
        for prior_song in self.playlist[-6:]:
            recent_artists.update(
                artist.strip().lower()
                for artist in prior_song.artists.split(";")
                if artist.strip()
            )

        available_idx = self.pool.available_indices()
        candidate_order = np.argsort(scores)[::-1][: min(48, len(scores))]
        if preferred_artists:
            preferred = {
                artist.strip().lower()
                for artist in preferred_artists
                if artist and artist.strip()
            }
            matched_locals = [
                int(local_idx)
                for local_idx in candidate_order
                if preferred
                and any(
                    artist_name in self.pool.get_song_info(int(available_idx[int(local_idx)])).artists.lower()
                    for artist_name in preferred
                )
            ]
            if matched_locals:
                candidate_order = np.array(matched_locals, dtype=int)
            elif require_artist_match:
                matching_all = [
                    local_idx
                    for local_idx in range(len(available_idx))
                    if any(
                        artist_name in self.pool.get_song_info(int(available_idx[local_idx])).artists.lower()
                        for artist_name in preferred
                    )
                ]
                if matching_all:
                    candidate_order = np.array(matching_all[:48], dtype=int)

        reranked_order = self._rerank_for_diversity(candidate_order, available_idx, scores)
        best_choice = None
        best_score = float("-inf")
        for local_idx in reranked_order:
            pool_idx = int(available_idx[local_idx])
            song = self.pool.get_song_info(pool_idx)
            adjusted = self._evaluate_candidate(
                song=song,
                base_score=float(scores[local_idx]),
                posterior_score=float(posterior_scores[local_idx]),
                prior_score=float(prior_scores[local_idx]),
                popularity_score=float(popularity_scores[local_idx]),
                discovery_score=float(discovery_scores[local_idx]),
                coherence_score=float(coherence_scores[local_idx]),
                recent_artists=recent_artists,
            )
            if adjusted > best_score:
                best_score = adjusted
                best_choice = (pool_idx, local_idx, song)

        if best_choice is None:
            return None

        pool_idx, local_best, song = best_choice

        self._current_song = song
        self._current_x = feat_matrix[local_best]
        self._current_feedback_action = None
        self.last_recommendation_score = float(best_score)
        return song

    def _rerank_for_diversity(
        self,
        candidate_order: np.ndarray,
        available_idx: np.ndarray,
        base_scores: np.ndarray,
        slate_size: int = 12,
    ) -> list[int]:
        if len(candidate_order) <= 1:
            return [int(idx) for idx in candidate_order]

        prompt_intent = getattr(self, "prompt_intent", None)
        allow_artist_run = bool(getattr(prompt_intent, "requested_single_artist", False))
        album_counts: dict[str, int] = {}
        artist_counts: dict[str, int] = {}
        for prior_song in self.playlist[-10:]:
            album_key = prior_song.album_name.strip().lower()
            if album_key:
                album_counts[album_key] = album_counts.get(album_key, 0) + 1
            for artist in prior_song.artists.split(";"):
                artist_key = artist.strip().lower()
                if artist_key:
                    artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1

        selected: list[int] = []
        remaining = [int(idx) for idx in candidate_order]
        while remaining and len(selected) < slate_size:
            best_idx = None
            best_score = float("-inf")
            for local_idx in remaining:
                pool_idx = int(available_idx[local_idx])
                song = self.pool.get_song_info(pool_idx)
                adjusted = float(base_scores[local_idx])
                album_key = song.album_name.strip().lower()
                if album_key:
                    adjusted -= 0.12 * album_counts.get(album_key, 0)
                    if album_counts.get(album_key, 0) >= 1:
                        adjusted -= 0.08
                if not allow_artist_run:
                    for artist in song.artists.split(";"):
                        artist_key = artist.strip().lower()
                        if artist_key:
                            adjusted -= 0.10 * artist_counts.get(artist_key, 0)
                            if artist_counts.get(artist_key, 0) >= 2:
                                adjusted -= 0.12
                if selected:
                    selected_songs = [
                        self.pool.get_song_info(int(available_idx[idx]))
                        for idx in selected[-4:]
                    ]
                    neighbor_penalty = 0.0
                    for prior in selected_songs:
                        feature_delta = np.mean(
                            [
                                abs(song.features[name] - prior.features[name])
                                for name in AUDIO_FEATURES
                            ]
                        )
                        if feature_delta < 0.08:
                            neighbor_penalty += 0.06
                        if song.album_name and prior.album_name and song.album_name.lower() == prior.album_name.lower():
                            neighbor_penalty += 0.10
                    adjusted -= neighbor_penalty
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = local_idx
            if best_idx is None:
                break
            selected.append(best_idx)
            chosen_song = self.pool.get_song_info(int(available_idx[best_idx]))
            album_key = chosen_song.album_name.strip().lower()
            if album_key:
                album_counts[album_key] = album_counts.get(album_key, 0) + 1
            if not allow_artist_run:
                for artist in chosen_song.artists.split(";"):
                    artist_key = artist.strip().lower()
                    if artist_key:
                        artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
            remaining.remove(best_idx)
        selected.extend(remaining)
        return selected

    def _prior_alignment_scores(self, feat_matrix: np.ndarray) -> np.ndarray:
        if self.model is None or not self.model.history:
            return np.zeros(feat_matrix.shape[0], dtype=np.float64)
        prior_mu = self.model.history[0].mu
        logits = feat_matrix @ prior_mu
        return 1.0 / (1.0 + np.exp(-logits))

    def _coherence_scores(self, feat_matrix: np.ndarray) -> np.ndarray:
        liked_songs = [
            song
            for song, action in zip(self.playlist, self.actions)
            if action == "play"
        ]
        if not liked_songs:
            return np.zeros(feat_matrix.shape[0], dtype=np.float64)

        liked_vectors = np.array(
            [
                [1.0, *[song.features[name] for name in AUDIO_FEATURES]]
                for song in liked_songs[-5:]
            ],
            dtype=np.float64,
        )
        centroid = liked_vectors.mean(axis=0)
        deltas = np.abs(feat_matrix - centroid)
        distances = deltas[:, 1:].mean(axis=1)
        return np.clip(1.0 - distances, 0.0, 1.0)

    def _evaluate_candidate(
        self,
        *,
        song: SongInfo,
        base_score: float,
        posterior_score: float,
        prior_score: float,
        popularity_score: float,
        discovery_score: float,
        coherence_score: float,
        recent_artists: set[str],
    ) -> float:
        adjusted = float(base_score)
        prompt_intent = getattr(self, "prompt_intent", None)
        exploration_weight = float(getattr(prompt_intent, "exploration_weight", 0.55) or 0.55)
        direct_match_weight = float(getattr(prompt_intent, "direct_match_weight", 0.45) or 0.45)
        if self.spec and self.spec.genres:
            normalized_genres = {genre.lower().replace("-", " ") for genre in self.spec.genres}
            song_genre = song.genre.lower().replace("-", " ")
            if song_genre in normalized_genres:
                adjusted += 0.06 + 0.06 * direct_match_weight
        excluded_artists = {
            artist.strip().lower()
            for artist in getattr(prompt_intent, "excluded_artists", []) or []
            if artist.strip()
        }
        if excluded_artists:
            song_artists_lower = {
                artist.strip().lower()
                for artist in song.artists.split(";")
                if artist.strip()
            }
            if song_artists_lower & excluded_artists:
                adjusted -= 0.7

        if prior_score < 0.42:
            adjusted -= 0.12
        if posterior_score < 0.42:
            adjusted -= 0.10
        if popularity_score < 0.32:
            adjusted -= 0.10
        if discovery_score < -0.08:
            adjusted -= 0.16
        if self.playlist and coherence_score < 0.35:
            adjusted -= 0.10
        if discovery_score > 0.18:
            adjusted += 0.10
        if song.prompt_score > 0.65:
            adjusted += 0.12 + 0.12 * direct_match_weight
        if song.novelty_score > 0.72:
            adjusted += 0.08 + 0.14 * exploration_weight
        if song.source_type == "related_artist_top":
            adjusted += 0.10 + 0.08 * exploration_weight
        elif song.source_type == "seed_artist_top":
            adjusted += 0.04 + 0.05 * direct_match_weight
        elif song.source_type == "history_anchor":
            adjusted -= 0.08 + 0.12 * exploration_weight
        if song.is_saved:
            adjusted -= 0.08 + 0.14 * exploration_weight
        if song.is_recent:
            adjusted -= 0.08 + 0.12 * exploration_weight
        if song.is_top_track:
            adjusted -= 0.06 + 0.10 * exploration_weight

        song_artists = {
            artist.strip().lower()
            for artist in song.artists.split(";")
            if artist.strip()
        }
        recent_familiar_count = sum(
            1
            for prior_song in self.playlist[-3:]
            if getattr(prior_song, "is_saved", False)
            or getattr(prior_song, "is_recent", False)
            or getattr(prior_song, "is_top_track", False)
        )
        if recent_familiar_count >= 1 and (
            song.is_saved or song.is_recent or song.is_top_track or song.source_type == "history_anchor"
        ):
            adjusted -= 0.08

        overlap = len(song_artists & recent_artists)
        if overlap:
            adjusted -= 0.10 * overlap

        repeat_count = sum(
            1
            for prior_song in self.playlist[-8:]
            if prior_song.artists.lower() == song.artists.lower()
        )
        adjusted -= 0.14 * repeat_count
        return adjusted

    def apply_feedback_to_current(self, played: bool) -> dict[str, float]:
        """Update the model for the active song without advancing playback."""
        if self.model is None or self._current_x is None or self._current_song is None:
            raise RuntimeError("No pending song to give feedback on.")

        x = self._current_x
        old_mu = self.model.mu.copy()
        y = 1 if played else 0
        self.model.update(x, y)
        delta = self.model.mu - old_mu

        self.model.snapshot(x=x, y=y)
        self._current_feedback_action = "play" if played else "skip"
        return {name: float(delta[idx]) for name, idx in FEATURE_INDEX.items()}

    def finalize_current_song(self, default_action: str | None = None) -> str | None:
        """Commit the active song to history and retire it from the pool."""
        if self._current_song is None:
            return None

        action = self._current_feedback_action or default_action
        if action in {"play", "skip"}:
            self.playlist.append(self._current_song)
            self.actions.append(action)
        self.pool.mark_song_used(
            pool_idx=getattr(self._current_song, "pool_idx", None),
            track_id=str(getattr(self._current_song, "track_id", "") or ""),
            track_name=str(getattr(self._current_song, "track_name", "") or ""),
            artists=str(getattr(self._current_song, "artists", "") or ""),
        )

        self._current_song = None
        self._current_x = None
        self._current_feedback_action = None
        return action

    def record_feedback(self, played: bool) -> dict[str, float]:
        """Update the model with the user's play/skip decision and advance."""
        delta = self.apply_feedback_to_current(played)
        self.finalize_current_song()
        return delta

    def current_feedback_action(self) -> str | None:
        return self._current_feedback_action

    def has_feedback_for_current(self) -> bool:
        return self._current_feedback_action in {"play", "skip"}

    def advance_without_feedback(self) -> None:
        """Advance past the current song without updating the posterior."""
        self.finalize_current_song(default_action=None)

    def run_interactive(self, prompt: str) -> None:
        """Full interactive CLI loop."""
        spec = self.start(prompt)

        print(f"\nParsed query:")
        print(f"  Genres:      {spec.genres or ['(any)']}")
        print(f"  Moods:       {spec.moods or ['(any)']}")
        if spec.constraints:
            print(f"  Constraints: ", end="")
            parts = [f"{k}=[{lo:.2f}, {hi:.2f}]" for k, (lo, hi) in spec.constraints.items()]
            print(", ".join(parts))
        print(f"\nCandidate pool: {self.pool.n_available:,} songs")
        if self.pool.n_available == 0:
            print("No songs matched your query. Try a broader prompt.")
            return
        print(f"Playlist length: {self.playlist_length}\n")

        for i in range(1, self.playlist_length + 1):
            song = self.recommend_next()
            if song is None:
                print("No more candidate songs available.")
                break

            print(f"--- Song {i}/{self.playlist_length} ---")
            print(f'  "{song.track_name}" by {song.artists} [{song.genre}]')
            print(
                f"  energy={song.features['energy']:.2f} | "
                f"danceability={song.features['danceability']:.2f} | "
                f"valence={song.features['valence']:.2f} | "
                f"tempo={song.raw_tempo:.0f} bpm"
            )

            action = self._prompt_action()
            if action == "q":
                print("\nEnding session early.")
                break

            played = action == "p"
            deltas = self.record_feedback(played)

            label = "Liked!" if played else "Skipped."
            top_shifts = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            shift_str = ", ".join(f"{n} {d:+.3f}" for n, d in top_shifts)
            print(f"  {label}  (weight shifts: {shift_str})\n")

        self._print_summary()

    def _prompt_action(self) -> str:
        while True:
            try:
                raw = input("  [p]lay / [s]kip / [q]uit? > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "q"
            if raw in ("p", "play"):
                return "p"
            if raw in ("s", "skip"):
                return "s"
            if raw in ("q", "quit", "exit"):
                return "q"
            print("  Please enter p, s, or q.")

    def _print_summary(self) -> None:
        if not self.playlist:
            return

        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)

        played = [s for s, a in zip(self.playlist, self.actions) if a == "play"]
        skipped = [s for s, a in zip(self.playlist, self.actions) if a == "skip"]
        print(f"Songs played: {len(played)}  |  Songs skipped: {len(skipped)}")

        if played:
            print("\nYour playlist:")
            for idx, song in enumerate(played, 1):
                print(f"  {idx}. \"{song.track_name}\" by {song.artists}")

        if self.model is not None:
            summary = self.model.get_summary()
            print("\nLearned preferences (posterior weights):")
            for name, w in sorted(
                summary.feature_weights.items(), key=lambda kv: abs(kv[1]), reverse=True
            ):
                direction = "+" if w > 0 else "-"
                print(f"  {name:20s}  {direction} {abs(w):.3f}")

        print("=" * 50)

        if self.analyze and self.model is not None and self.model.history:
            print(f"\nGenerating diagnostic plots to {self.output_dir}/ ...")
            paths = generate_all_diagnostics(
                self.model.history, output_dir=self.output_dir
            )
            for p in paths:
                print(f"  Saved: {p}")
            print()
