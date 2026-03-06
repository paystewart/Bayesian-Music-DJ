from __future__ import annotations

from pathlib import Path

import numpy as np

from music_query_parser.parser import MusicQueryParser, QuerySpec

from .diagnostics import generate_all_diagnostics
from .model import FEATURE_INDEX, BayesianLogisticRegression
from .song_pool import SongInfo, SongPool

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

    def start(self, prompt: str) -> QuerySpec:
        """Parse the prompt, filter the song pool, and initialise the prior."""
        self.spec = self.parser.parse(prompt)

        self.pool.filter_by_genres(self.spec.genres)

        constraints = dict(self.spec.constraints)
        self.model = BayesianLogisticRegression.from_constraints(constraints)

        self.model.snapshot()

        return self.spec

    def recommend_next(self) -> SongInfo | None:
        """Use Thompson sampling to pick the next song."""
        if self.model is None:
            raise RuntimeError("Call start() before recommending songs.")
        if self.pool.n_available == 0:
            return None

        feat_matrix = self.pool.get_feature_matrix()
        posterior_scores = self.model.predict_proba_posterior(feat_matrix)
        thompson_scores = self.model.thompson_sample_scores(feat_matrix)
        popularity_scores = self.pool.get_popularity_scores()
        scores = (
            0.52 * thompson_scores
            + 0.28 * posterior_scores
            + 0.20 * popularity_scores
        )

        available_idx = self.pool.available_indices()
        local_best = int(np.argmax(scores))
        pool_idx = int(available_idx[local_best])

        song = self.pool.get_song_info(pool_idx)
        self._current_song = song
        self._current_x = feat_matrix[local_best]
        return song

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
