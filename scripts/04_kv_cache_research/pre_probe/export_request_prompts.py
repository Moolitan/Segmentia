from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = BASE_DIR.parent.parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "03_14B_anthropic_3"
    / "slack_launch_pack"
    / "multiturn_sequence_traces.json"
)
DEFAULT_OUTPUT_DIR = BASE_DIR / "prompts" / "multiturn_sequence_traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export each request_prompt_text in llm_calls to its own txt file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for txt files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Wrap long lines to this width. Use 0 to keep original lines.",
    )
    return parser.parse_args()


def load_llm_calls(json_path: Path) -> list[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level object in {json_path}, got {type(data).__name__}.")

    llm_calls = data.get("llm_calls")
    if not isinstance(llm_calls, list):
        raise ValueError(f"Expected llm_calls to be a list in {json_path}.")

    return llm_calls


def wrap_line(line: str, width: int) -> str:
    if width <= 0 or len(line) <= width or not line.strip():
        return line

    indent = line[: len(line) - len(line.lstrip(" "))]
    content = line[len(indent) :]

    wrapper = textwrap.TextWrapper(
        width=max(width, len(indent) + 1),
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=True,
    )
    return wrapper.fill(content)


def normalize_text(text: str, width: int) -> str:
    if width <= 0:
        normalized = text
    else:
        normalized = "\n".join(wrap_line(line, width) for line in text.splitlines())

    if text.endswith("\n"):
        return normalized + "\n"
    return normalized


def export_prompts(llm_calls: list[dict], output_dir: Path, width: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    for old_file in output_dir.glob("*.txt"):
        old_file.unlink()

    digits = max(3, len(str(len(llm_calls))))
    exported = 0

    for index, item in enumerate(llm_calls, start=1):
        prompt_text = item.get("request_prompt_text")
        if not isinstance(prompt_text, str):
            raise ValueError(
                f"llm_calls[{index - 1}].request_prompt_text is not a string: "
                f"{type(prompt_text).__name__}"
            )

        file_path = output_dir / f"{index:0{digits}d}.txt"
        file_path.write_text(normalize_text(prompt_text, width), encoding="utf-8")
        exported += 1

    return exported


def main() -> None:
    args = parse_args()
    llm_calls = load_llm_calls(args.input)
    exported = export_prompts(llm_calls, args.output_dir, args.width)
    print(f"Exported {exported} prompt files to {args.output_dir}")


if __name__ == "__main__":
    main()
