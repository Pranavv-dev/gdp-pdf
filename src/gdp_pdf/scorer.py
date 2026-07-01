"""Per-criterion model-graded scorer.

How grading works:

- One judge call *per criterion*. The judge sees only the response and that
  single criterion — never the original task prompt or the PDF.
- Structured output is requested with a JSON schema whose `score` is the
  string enum `["0", "1"]`.
- Output is parsed strictly: markdown fences stripped, strict JSON parse,
  schema-validated (`score` a number or string, `rationale` a string), score
  coerced to a decimal. A criterion passes iff `score.round(4) >= 1.0`.
- A judge call whose output fails to parse is retried up to `MAX_RETRIES`
  more times (5 attempts total). If all attempts fail the criterion is
  *unscored*: it is excluded from the `mean_criteria` denominator, and the
  whole response is excluded from `all_pass` (a response only counts toward
  the headline metric when every criterion was scored). Exclusion is
  signalled with NaN values, which Inspect reports as unscored samples
  rather than counting them as failures.

Each sample yields two values that map directly to the leaderboard numbers:

- `all_pass`: 1.0 iff every criterion is satisfied (the headline Pass@1),
- `mean_criteria`: fraction of scored criteria satisfied ("Average Response
  Pass Rate").
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, TypedDict

from inspect_ai.model import GenerateConfig, Model, ResponseSchema, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import JSONSchema

# Additional attempts after the first when the judge output fails to parse.
MAX_RETRIES = 4

# Judge prompt template. The `{{ ... }}` variables are substituted with the
# raw response and criterion strings.
JUDGE_PROMPT_TEMPLATE = """Read this json object:
{
  response: {{ rubric.response_to_grade }},
  criteria: {{ rubric.criterion }},
}

Score the response based on the criteria.
If the response perfectly satisfies the criteria, give it a score of "1".
If it does not satisfy the criteria at all, give it a score of "0".
Return the score and rationale in a JSON object with two keys, 'score' and 'rationale'.
'score' should be a string that's either "0" or "1".
'rationale' should be a description of why you gave the response that score.

Here are some examples of how to score a response:
{ score: "1", rationale: "The response is a perfect match for the criteria." }
{ score: "0", rationale: "The response does not satisfy the criteria at all." }
Unlike these examples, when you create a rationale, be specific and detailed. Give reasons specific to this response and criteria. Don't be generic."""

# Structured output schema constraining the judge's verdict.
RESPONSE_SCHEMA = ResponseSchema(
    name="individual_criteria_score",
    json_schema=JSONSchema(
        type="object",
        required=["rationale", "score"],
        properties={
            "rationale": JSONSchema(
                type="string",
                description="Text describing the evaluation criterion",
            ),
            "score": JSONSchema(
                type="string",
                enum=["0", "1"],
                description="Score value: '0' (does not satisfy) or '1' (perfectly satisfies)",
            ),
        },
        additionalProperties=False,
    ),
)

# Sentinel for "excluded from the metric" — Inspect drops NaN-valued entries
# from metric aggregation and reports them as unscored samples.
_EXCLUDED = float("nan")


class ParseError(ValueError):
    """Judge output failed strict parsing/validation."""


class CriterionVerdict(TypedDict):
    score: Decimal
    rationale: str


@scorer(metrics={"all_pass": [mean()], "mean_criteria": [mean()]})
def rubric_scorer(judge_model: str) -> Scorer:
    # JUDGE_BASE_URL / JUDGE_API_KEY env vars let the judge use different
    # credentials than the model under test when both share a provider (e.g.
    # both `google/`). If unset, the provider's default credentials apply.
    grader = get_model(
        judge_model,
        base_url=os.environ.get("JUDGE_BASE_URL"),
        api_key=os.environ.get("JUDGE_API_KEY"),
    )

    async def score(state: TaskState, target: Target) -> Score:
        rubrics: list[dict[str, str]] = state.metadata.get("rubrics", [])
        answer = state.output.completion

        verdicts = await asyncio.gather(
            *(_score_criterion(grader, answer, rubric["criterion"]) for rubric in rubrics)
        )

        grades = [
            {
                "rubric_id": rubric["id"],
                "result": (
                    "unscored"
                    if verdict is None
                    else "pass"
                    if _criterion_passes(verdict["score"])
                    else "fail"
                ),
                "score": None if verdict is None else str(verdict["score"]),
                "rationale": None if verdict is None else verdict["rationale"],
            }
            for rubric, verdict in zip(rubrics, verdicts)
        ]

        scored = [grade for grade in grades if grade["result"] != "unscored"]
        passed = sum(grade["result"] == "pass" for grade in grades)

        # A response missing any criterion score is excluded from `all_pass`
        # entirely; the pass-rate denominator counts only scored criteria.
        fully_scored = bool(rubrics) and len(scored) == len(rubrics)
        return Score(
            value={
                "all_pass": (1.0 if passed == len(rubrics) else 0.0)
                if fully_scored
                else _EXCLUDED,
                "mean_criteria": passed / len(scored) if scored else _EXCLUDED,
            },
            answer=answer,
            explanation=json.dumps(grades, indent=2),
            metadata={"rubric_grades": grades},
        )

    return score


async def _score_criterion(
    grader: Model, response_text: str, criterion: str
) -> CriterionVerdict | None:
    """Judge one criterion, retrying parse failures. None = unscored."""
    prompt = _judge_prompt(response_text, criterion)
    for _ in range(MAX_RETRIES + 1):
        result = await grader.generate(
            prompt, config=GenerateConfig(response_schema=RESPONSE_SCHEMA)
        )
        try:
            return _parse_criterion_result(result.completion)
        except ParseError:
            continue
    return None


def _judge_prompt(response_text: str, criterion: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.replace(
        "{{ rubric.response_to_grade }}", response_text
    ).replace("{{ rubric.criterion }}", criterion)


def _parse_criterion_result(raw_text: str) -> CriterionVerdict:
    """Strict parse of the judge output.

    Strips markdown fences, requires valid JSON, and validates that `score` is
    a number or string coercible to a decimal and `rationale` is a string.
    Raises `ParseError` on any violation so the caller can retry.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"\A```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\Z", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ParseError("Judge output must be a JSON object")
    if "score" not in payload or "rationale" not in payload:
        raise ParseError("Judge output must contain 'score' and 'rationale'")
    raw_score: Any = payload["score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float, str)):
        raise ParseError("'score' must be a number or string")
    if not isinstance(payload["rationale"], str):
        raise ParseError("'rationale' must be a string")
    try:
        score = Decimal(str(raw_score).strip())
    except InvalidOperation as e:
        raise ParseError(f"'score' is not a valid decimal: {raw_score!r}") from e

    return CriterionVerdict(score=score, rationale=payload["rationale"])


def _criterion_passes(score: Decimal) -> bool:
    """Pass condition: `score.round(4) >= 1.0` (half-up)."""
    return score.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1
