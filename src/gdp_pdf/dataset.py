"""Load GDP.pdf tasks as Inspect `Sample`s.

Each task is a prompt plus one or more attachments (a PDF, sometimes an image)
and a list of rubric criteria. The prompt + attachments become a multimodal
user message; the rubric criteria ride along in `Sample.metadata` for the
scorer to grade against.

Two sources are supported: the Hugging Face dataset (`surgeai/GDP.pdf`,
parquet + PDFs via Git LFS) and a local YAML task pack for offline use. Provide
exactly one.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageUser,
    ContentDocument,
    ContentImage,
    ContentText,
)

# Anchored to the repo root (not the cwd): `inspect eval` chdirs into the task
# file's directory, which would otherwise split the cache across two locations.
PDF_CACHE_DIR = Path(__file__).resolve().parents[2] / "pdf_cache"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
# Matches the HF rubric columns, e.g. "rubric - 12. criterion".
_RUBRIC_RE = re.compile(r"rubric\s*-\s*(\d+)\.\s*criterion", re.IGNORECASE)


def load_samples(
    hf_repo: str | None = None,
    split: str = "test",
    task_pack: str | None = None,
    limit: int | None = None,
) -> MemoryDataset:
    """Build the dataset from whichever source is configured.

    The prompt and its attachments (PDF, sometimes an image) become a single
    multimodal user message; provide exactly one of `hf_repo` or `task_pack`.
    """
    if task_pack is not None:
        samples = _samples_from_task_pack(Path(task_pack))
        if limit is not None:
            samples = samples[:limit]
    elif hf_repo is not None:
        # Pass the limit down so we stop before downloading unneeded PDFs.
        samples = _samples_from_hf(hf_repo, split, limit)
    else:
        raise ValueError("provide either `hf_repo` or `task_pack`")
    return MemoryDataset(samples=samples, name="gdp_pdf")


def _make_sample(
    task_id: str,
    prompt: str,
    attachments: list[Path],
    rubrics: list[dict[str, str]],
    metadata: dict | None = None,
) -> Sample:
    content = [ContentText(text=prompt)] + [
        _attachment_content(p) for p in attachments
    ]
    return Sample(
        id=str(task_id),
        input=[ChatMessageUser(content=content)],
        # The judge sees only the response and one criterion (never the prompt
        # or attachments); the plain prompt copy is kept for log inspection.
        metadata={"prompt": prompt, "rubrics": rubrics, **(metadata or {})},
    )


def _attachment_content(path: Path) -> ContentDocument | ContentImage:
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        return ContentImage(image=str(path))
    return ContentDocument(document=str(path))


def _samples_from_hf(
    hf_repo: str,
    split: str,
    limit: int | None = None,
) -> list[Sample]:
    """Load + validate tasks from the Hugging Face parquet dataset.

    Reads the split, gathers the `rubric - N. criterion` columns, and resolves
    each `pdf_path` to a local file downloaded into `PDF_CACHE_DIR`. When `limit`
    is set, stops after that many rows so we don't download PDFs we won't use.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    rows = load_dataset(hf_repo, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    samples: list[Sample] = []
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id") or row.get("id") or index)
        pdf_rel = row["pdf_path"]
        local_pdf = Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=pdf_rel,
                repo_type="dataset",
                cache_dir=str(PDF_CACHE_DIR),
            )
        )
        samples.append(
            _make_sample(
                task_id,
                row["prompt"],
                [local_pdf],
                _rubrics_from_row(row),
                {"pdf_path": pdf_rel, "source": hf_repo},
            )
        )
    return samples


def _rubrics_from_row(row: dict) -> list[dict[str, str]]:
    """Collect non-empty `rubric - N. criterion` columns, ordered by N."""
    found: list[tuple[int, str]] = []
    for column, value in row.items():
        match = _RUBRIC_RE.fullmatch(str(column).strip())
        if match and value not in (None, ""):
            found.append((int(match.group(1)), str(value)))
    found.sort(key=lambda pair: pair[0])
    return [{"id": f"r{n}", "criterion": text} for n, text in found]


def _samples_from_task_pack(root: Path) -> list[Sample]:
    """Load + validate tasks from a local YAML task pack (offline format)."""
    root = Path(root)
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    if not manifest or "tasks" not in manifest:
        raise ValueError(f"task pack {root} has no `tasks` in manifest.yaml")

    samples: list[Sample] = []
    for rel in manifest["tasks"]:
        spec = yaml.safe_load((root / rel).read_text())
        attachments = [root / a for a in spec.get("attachments", [])]
        for path in attachments:
            if not path.exists():
                raise FileNotFoundError(f"missing attachment {path} for task {spec['id']}")
        rubrics = [
            {"id": str(r["id"]), "criterion": r["criterion"]} for r in spec.get("rubrics", [])
        ]
        samples.append(
            _make_sample(
                str(spec["id"]),
                spec["prompt"],
                attachments,
                rubrics,
                spec.get("metadata", {}),
            )
        )
    return samples
