import argparse
from pathlib import Path
import os

from openai import OpenAI
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "prompts" / "system_prompt.md"

load_dotenv(ROOT / ".env")


def load_system_prompt(path: Path) -> str:
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing prompt file: {path}") from exc

    if not prompt:
        raise RuntimeError(f"Prompt file is empty: {path}")

    return prompt


def create_client() -> OpenAI:
    base_url = os.getenv("LLM_GATEWAY_URL")
    api_key = os.getenv("GATEWAY_API_KEY")

    if not base_url or not api_key:
        raise RuntimeError("LLM_GATEWAY_URL and GATEWAY_API_KEY must be set")

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=30,
    )


def generate_limerick(topic: str, model: str, system_prompt: str) -> str:
    client = create_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": topic},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned no message content")
    return content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a limerick about a topic using the configured system prompt."
    )
    parser.add_argument(
        "--model", required=True, help="Model name to send to the LLM gateway"
    )
    parser.add_argument(
        "--topic", required=True, help="Topic for the limerick"
    )
    args = parser.parse_args()

    system_prompt = load_system_prompt(PROMPT_PATH)
    print(generate_limerick(args.topic, args.model, system_prompt))


if __name__ == "__main__":
    main()
