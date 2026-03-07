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
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV,
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
        self.pool = SongPool(csv_path)
        self.playlist_length = playlist_length
        self.analyze = analyze
        self.output_dir = Path(output_dir)

        self.model: BayesianLogisticRegression | None = None
        self.spec: QuerySpec | None = None
        self.playlist: list[SongInfo] = []
        self.actions: list[str] = []
        self._current_song: SongInfo | None = None
        self._current_x: np.ndarray | None = None
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
        preference_scores = self.pool.get_external_bias_scores()
        prior_scores = self._prior_alignment_scores(feat_matrix)
        coherence_scores = self._coherence_scores(feat_matrix)
        scores = (
            0.22 * thompson_scores
            + 0.22 * posterior_scores
            + 0.18 * prior_scores
            + 0.16 * popularity_scores
            + 0.12 * preference_scores
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

        best_choice = None
        best_score = float("-inf")
        for local_idx in candidate_order:
            pool_idx = int(available_idx[local_idx])
            song = self.pool.get_song_info(pool_idx)
            adjusted = self._evaluate_candidate(
                song=song,
                base_score=float(scores[local_idx]),
                posterior_score=float(posterior_scores[local_idx]),
                prior_score=float(prior_scores[local_idx]),
                popularity_score=float(popularity_scores[local_idx]),
                preference_score=float(preference_scores[local_idx]),
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
        self.last_recommendation_score = float(best_score)
        return song

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
        preference_score: float,
        coherence_score: float,
        recent_artists: set[str],
    ) -> float:
        adjusted = float(base_score)
        if self.spec and self.spec.genres:
            normalized_genres = {genre.lower().replace("-", " ") for genre in self.spec.genres}
            song_genre = song.genre.lower().replace("-", " ")
            if song_genre in normalized_genres:
                adjusted += 0.08

        if prior_score < 0.42:
            adjusted -= 0.22
        if posterior_score < 0.42:
            adjusted -= 0.16
        if popularity_score < 0.32:
            adjusted -= 0.12
        if self.playlist and coherence_score < 0.35:
            adjusted -= 0.18
        if preference_score > 0.72:
            adjusted += 0.05

        song_artists = {
            artist.strip().lower()
            for artist in song.artists.split(";")
            if artist.strip()
        }
        overlap = len(song_artists & recent_artists)
        if overlap:
            adjusted -= 0.12 * overlap

        repeat_count = sum(
            1
            for prior_song in self.playlist[-8:]
            if prior_song.artists.lower() == song.artists.lower()
        )
        adjusted -= 0.18 * repeat_count
        return adjusted

    def record_feedback(self, played: bool) -> dict[str, float]:
        """Update the model with the user's play/skip decision.

        Returns a dict mapping feature names to their weight *change*
        so the caller can display the posterior shift.
        """
        if self.model is None or self._current_x is None or self._current_song is None:
            raise RuntimeError("No pending song to give feedback on.")

        x = self._current_x
        old_mu = self.model.mu.copy()
        y = 1 if played else 0
        self.model.update(x, y)
        delta = self.model.mu - old_mu

        self.model.snapshot(x=x, y=y)

        self.playlist.append(self._current_song)
        self.actions.append("play" if played else "skip")
        self.pool.mark_used(self._current_song.pool_idx)

        self._current_song = None
        self._current_x = None

        return {name: float(delta[idx]) for name, idx in FEATURE_INDEX.items()}

    def advance_without_feedback(self) -> None:
        """Advance past the current song without updating the posterior."""
        if self._current_song is None:
            return

        self.pool.mark_used(self._current_song.pool_idx)
        self._current_song = None
        self._current_x = None

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
