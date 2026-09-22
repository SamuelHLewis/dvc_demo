import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models import infer_model
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "params.yaml"
PROMPT_PATH = ROOT / "prompts" / "judge_prompt.md"
RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"
SCORES_PATH = Path(__file__).resolve().parent / "evaluation_results.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_client, load_system_prompt


class JudgeOutput(BaseModel):
    score: str
    reason: str


class LimerickJudge(Evaluator[dict[str, str], str, None]):
    def __init__(self, rubric: str, model: Model) -> None:
        self._agent = Agent(
            model,
            system_prompt=rubric,
            output_type=JudgeOutput,
        )

    def get_default_evaluation_name(self) -> str:
        return "limerick_judge"

    async def evaluate(self, ctx: EvaluatorContext[dict[str, str], str, None]) -> EvaluationReason:
        result = (await self._agent.run(f"Topic: {ctx.inputs['topic']}\n\nLimerick:\n{ctx.output}")).output

        if result.score not in {"bad", "acceptable", "inspired"}:
            raise RuntimeError(f"Judge returned an invalid score: {result.score!r}")

        return EvaluationReason(value=result.score, reason=result.reason)


def load_judge_model(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing params file: {path}") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition(":")
        if separator and key.strip() == "judge_model":
            model = value.strip()
            if model:
                return model
            break

    raise RuntimeError(f"judge_model is missing from params file: {path}")


def load_results(path: Path) -> list[dict[str, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing eval results file: {path}") from exc

    results: list[dict[str, str]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON on line {line_number} of {path}") from exc

        topic = record.get("topic")
        limerick = record.get("limerick")
        if not isinstance(topic, str) or not isinstance(limerick, str):
            raise RuntimeError(f"Each result must contain string 'topic' and 'limerick' fields: line {line_number}")

        results.append({"topic": topic, "limerick": limerick})

    if not results:
        raise RuntimeError(f"No eval results found in {path}")

    return results


def build_dataset(results: list[dict[str, str]], rubric: str, model_name: str) -> Dataset[dict[str, str], str, None]:
    client = create_client()
    api_key = client.api_key if isinstance(client.api_key, str) else os.getenv("GATEWAY_API_KEY")
    if not api_key:
        raise RuntimeError("GATEWAY_API_KEY must be set")

    judge_model = infer_model(
        f"openai:{model_name}",
        provider_factory=lambda _provider_name: OpenAIProvider(
            base_url=str(client.base_url),
            api_key=api_key,
        ),
    )

    cases = [
        Case(
            name=f"topic_{index}",
            inputs={"topic": result["topic"]},
            expected_output=None,
            metadata=None,
        )
        for index, result in enumerate(results, start=1)
    ]

    return Dataset(
        name="limerick_judge",
        cases=cases,
        evaluators=[LimerickJudge(rubric=rubric, model=judge_model)],
    )


def write_scores(path: Path, scored_results: list[dict[str, str]]) -> None:
    lines = [json.dumps(result, ensure_ascii=False) for result in scored_results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run() -> None:
    parser = argparse.ArgumentParser(
        description="Judge generated limericks in evals/results.jsonl with pydantic-evals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_PATH,
        help="Path to the JSONL file produced by evals/run_eval.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCORES_PATH,
        help="Path to write the judged eval results as JSONL",
    )
    args = parser.parse_args()

    rubric = load_system_prompt(PROMPT_PATH)
    model_name = load_judge_model(PARAMS_PATH)
    results = load_results(args.input)
    dataset = build_dataset(results, rubric, model_name)

    limericks_by_topic = {result["topic"]: result["limerick"] for result in results}

    def get_limerick(inputs: dict[str, str]) -> str:
        return limericks_by_topic[inputs["topic"]]

    report = await dataset.evaluate(get_limerick)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    scored_results: list[dict[str, str]] = []
    for case in report.cases:
        score_result = case.labels.get("limerick_judge")
        if score_result is None:
            raise RuntimeError(f"Missing limerick score for case {case.name}")

        scored_results.append(
            {
                "topic": case.inputs["topic"],
                "score": score_result.value,
                "reason": score_result.reason,
            }
        )

    write_scores(args.output, scored_results)
    print(f"Wrote judged results to {args.output}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
