from __future__ import annotations

import argparse
import json

from .parser import MusicQueryParser

EXAMPLE_PROMPTS = [
    "Make a chill indie playlist like Phoebe Bridgers, 2018-2022, not too upbeat",
    "Workout EDM, high energy, 120-140 bpm",
    "Start with 'Blinding Lights' then similar party pop",
    "Sad acoustic songs, 15 tracks, 90-110 bpm",
    "Focus music for coding, low energy, mostly instrumental vibes",
    "Happy latin dance",
    "Dark dreamy synthwave from the 80s",
    "Underground hip hop, deep cuts, low popularity",
    "Epic orchestral, no vocals, high energy",
    "Songs like Radiohead and Tame Impala, moody alternative",
    "Popular top hits from 2023, danceable pop",
    "Smooth jazz by Miles Davis, peaceful, low tempo",
]


def run_examples(parser: MusicQueryParser) -> None:
    print("=== Unit-test-like Examples ===")
    for idx, prompt in enumerate(EXAMPLE_PROMPTS, start=1):
        spec = parser.parse(prompt)
        print(f"\n[{idx}] Prompt: {prompt}")
        print(json.dumps(spec.to_dict(), indent=2))


def run_cli_loop(parser: MusicQueryParser) -> None:
    print("Enter a prompt to parse (or 'quit'):")
    while True:
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit"}:
            return
        spec = parser.parse(prompt)
        print(json.dumps(spec.to_dict(), indent=2))


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Parse music prompts into structured query specs.")
    arg_parser.add_argument("--prompt", type=str, default=None, help="Single prompt to parse.")
    arg_parser.add_argument(
        "--examples",
        action="store_true",
        help="Run built-in examples and print parsed output.",
    )
    arg_parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    arg_parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/music_query_parser",
        help="Cache directory for model and precomputed label embeddings.",
    )
    args = arg_parser.parse_args()

    parser = MusicQueryParser(model_name=args.model_name, cache_dir=args.cache_dir)
    if args.examples:
        run_examples(parser)
        return
    if args.prompt:
        spec = parser.parse(args.prompt)
        print(json.dumps(spec.to_dict(), indent=2))
        return
    run_cli_loop(parser)


if __name__ == "__main__":
    main()
