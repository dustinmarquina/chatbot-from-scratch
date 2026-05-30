from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.infer_track_b import load_track_b
from src.track_a import load_track_a


DEFAULT_PROMPTS = [
    "Hello, how are you today?",
    "What do you like about machine learning?",
    "Tell me a short joke.",
    "How can I stay productive while studying?",
    "Explain attention in one sentence.",
]


def compare_models(
    prompts: list[str],
    track_a: Any,
    track_b: Any,
    output_path: str | Path,
) -> list[dict[str, str]]:
    rows = []
    for prompt in prompts:
        rows.append(
            {
                "prompt": prompt,
                "track_a_reply": track_a.generate_reply([], prompt),
                "track_b_reply": track_b.generate_reply([], prompt),
                "note": "",
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    return rows


def main(
    checkpoint_path: str = "artifacts/checkpoints/track_b_checkpoint.pt",
    output_path: str = "artifacts/comparison.json",
) -> list[dict[str, str]]:
    track_a = load_track_a()
    track_b = load_track_b(checkpoint_path)
    rows = compare_models(DEFAULT_PROMPTS, track_a, track_b, output_path)
    for row in rows:
        print(f"Prompt: {row['prompt']}")
        print(f"Track A: {row['track_a_reply']}")
        print(f"Track B: {row['track_b_reply']}")
        print("-" * 40)
    return rows


if __name__ == "__main__":
    main()
