from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.infer_track_c import load_track_c
from src.track_a import load_track_a


DEFAULT_PROMPTS = [
    "Hello, how are you today?",
    "What do you like about machine learning?",
    "Tell me a short joke.",
    "How can I stay productive while studying?",
    "Explain attention in one sentence.",
]


def compare_track_c(
    prompts: list[str],
    track_a: Any,
    track_c: Any,
    output_path: str | Path,
) -> list[dict[str, str]]:
    rows = []
    for prompt in prompts:
        rows.append(
            {
                "prompt": prompt,
                "track_a_reply": track_a.generate_reply([], prompt),
                "track_c_reply": track_c.generate_reply([], prompt),
                "note": "",
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2))
    return rows


def main(
    model_dir: str = "artifacts/track_c_distilgpt2",
    output_path: str = "artifacts/comparison_track_c.json",
) -> list[dict[str, str]]:
    track_a = load_track_a()
    track_c = load_track_c(model_dir)
    rows = compare_track_c(DEFAULT_PROMPTS, track_a, track_c, output_path)
    for row in rows:
        print(f"Prompt: {row['prompt']}")
        print(f"Track A: {row['track_a_reply']}")
        print(f"Track C: {row['track_c_reply']}")
        print("-" * 40)
    return rows


if __name__ == "__main__":
    main()
