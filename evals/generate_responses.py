import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
TOPICS_PATH = ROOT / "topics.txt"
PARAMS_PATH = ROOT / "params.yaml"
RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import PROMPT_PATH, generate_limerick, load_system_prompt


def load_topics(path: Path) -> list[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing topics file: {path}") from exc


def load_app_model(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing params file: {path}") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if separator and key.strip() == "app_model":
            model = value.strip()
            if model:
                return model
            break

    raise RuntimeError(f"app_model is missing from params file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate limericks for each topic and save them to evals/results.jsonl."
    )
    parser.add_argument(
        "--model",
        default=load_app_model(PARAMS_PATH),
        help="Model name to send to the LLM gateway (defaults to params.yaml app_model)",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        default=TOPICS_PATH,
        help="Path to the topics file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the JSONL output file",
    )
    args = parser.parse_args()

    topics = load_topics(args.topics_file)
    system_prompt = load_system_prompt(PROMPT_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        for topic in topics:
            limerick = generate_limerick(topic, args.model, system_prompt)
            json.dump({"topic": topic, "limerick": limerick}, output_file, ensure_ascii=False)
            output_file.write("\n")


if __name__ == "__main__":
    main()
