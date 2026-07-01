"""Offline smoke tests: no network, no API keys, no model calls.

Exercises the two pieces of custom logic that Inspect doesn't provide: loading
GDP.pdf tasks into `Sample`s (multimodal content + rubric metadata) and the
strict parsing behind the per-criterion scorer.

Run with: uv run python tests/test_offline.py
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from inspect_ai.model import ContentDocument, ContentText

from gdp_pdf.dataset import load_samples
from gdp_pdf.scorer import (
    ParseError,
    _criterion_passes,
    _judge_prompt,
    _parse_criterion_result,
)

REPO = Path(__file__).resolve().parent.parent
PACK = REPO / "task_packs" / "sample_pdf"


def test_load_task_pack_samples() -> None:
    samples = {s.id: s for s in load_samples(task_pack=str(PACK)).samples}
    assert set(samples) == {"sample_001", "sample_002"}
    for sample in samples.values():
        assert sample.metadata["prompt"].strip()
        rubrics = sample.metadata["rubrics"]
        assert rubrics and all(r["id"] and r["criterion"] for r in rubrics)
    assert [r["id"] for r in samples["sample_002"].metadata["rubrics"]] == [
        "r1",
        "r2",
        "r3",
        "r4",
    ]


def test_sample_multimodal_content() -> None:
    sample = load_samples(task_pack=str(PACK)).samples[0]
    content = sample.input[0].content
    assert isinstance(content[0], ContentText)
    assert isinstance(content[1], ContentDocument)
    assert content[1].mime_type == "application/pdf"


def test_limit() -> None:
    assert len(load_samples(task_pack=str(PACK), limit=1).samples) == 1


def test_parse_criterion_result() -> None:
    # Canonical shape: {"score": "1", "rationale": "..."}.
    verdict = _parse_criterion_result('{"score": "1", "rationale": "ok"}')
    assert verdict["score"] == Decimal("1")
    assert verdict["rationale"] == "ok"

    # Markdown fences are stripped.
    fenced = '```json\n{"score": "0", "rationale": "no"}\n```'
    assert _parse_criterion_result(fenced)["score"] == Decimal("0")

    # Numeric scores also validate (parsing allows number|string).
    assert _parse_criterion_result('{"score": 1, "rationale": "ok"}')["score"] == Decimal("1")

    # Violations raise ParseError so the caller retries.
    for bad in [
        "sorry, I can't help with that",  # not JSON
        '[{"score": "1", "rationale": "ok"}]',  # not an object
        '{"score": "1"}',  # missing rationale
        '{"rationale": "ok"}',  # missing score
        '{"score": true, "rationale": "ok"}',  # score not number|string
        '{"score": "maybe", "rationale": "ok"}',  # score not a decimal
        '{"score": "1", "rationale": 5}',  # rationale not a string
    ]:
        try:
            _parse_criterion_result(bad)
        except ParseError:
            pass
        else:
            raise AssertionError(f"expected ParseError for {bad!r}")


def test_criterion_passes_threshold() -> None:
    # Pass condition: score.round(4) >= 1.0 — fractional scores fail.
    assert _criterion_passes(Decimal("1")) is True
    assert _criterion_passes(Decimal("1.0")) is True
    assert _criterion_passes(Decimal("0.99999"))  # rounds half-up to 1.0
    assert _criterion_passes(Decimal("0.9999")) is False
    assert _criterion_passes(Decimal("0.75")) is False
    assert _criterion_passes(Decimal("0.5")) is False
    assert _criterion_passes(Decimal("0")) is False


def test_judge_prompt_template() -> None:
    prompt = _judge_prompt("the response", "the criterion")
    assert prompt.startswith("Read this json object:")
    assert "response: the response," in prompt
    assert "criteria: the criterion," in prompt
    # Judge input never includes the task prompt or attachments.
    assert "{{" not in prompt


def _run_all() -> None:
    test_load_task_pack_samples()
    test_sample_multimodal_content()
    test_limit()
    test_parse_criterion_result()
    test_criterion_passes_threshold()
    test_judge_prompt_template()
    print("all offline tests passed")


if __name__ == "__main__":
    _run_all()
