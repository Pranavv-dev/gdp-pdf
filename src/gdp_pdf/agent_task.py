"""GDP.pdf with a CLI coding agent instead of a single completion.

Added for parity with the Harbor adapter. Each sample runs in a Docker sandbox
built from the same Dockerfile as the Harbor task (minus the verifier's judge
client). The PDF is copied to /workspace/document.pdf, the agent gets the
Harbor task's instruction text, and whatever it writes to /workspace/answer.md
becomes the response that `rubric_scorer` grades, unchanged.

    uv run inspect eval src/gdp_pdf/agent_task.py@gdp_pdf_agent \
      --model anthropic/claude-... -T judge_model=... -T agent_version=2.x.y
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentText, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox
from inspect_swe import claude_code, codex_cli

from gdp_pdf.dataset import load_samples
from gdp_pdf.scorer import rubric_scorer
from gdp_pdf.task import DEFAULT_JUDGE_MODEL

# Surge's repo is gone from Hugging Face; this is the pinned copy the Harbor
# adapter uses (a "Duplicate" of surgeai/GDP.pdf taken 2026-09-08).
DEFAULT_HF_REPO = "cloudybean/GDP.pdf"
DEFAULT_HF_REVISION = "1b73b865da6be76dca2e4f3ad66a53d6709ac9f5"

SANDBOX_DIR = Path(__file__).parent / "agent_sandbox"
DOCUMENT_PATH = "/workspace/document.pdf"
ANSWER_PATH = "/workspace/answer.md"

# Same text as the Harbor adapter's task-template/instruction.md.
INSTRUCTION = """{prompt}

---

The attached document is at `/workspace/document.pdf`.

Write your complete final response to `/workspace/answer.md`. Only the contents of that file will be read.
"""


def _agent_sample(sample: Sample) -> Sample:
    """Swap the inline PDF for a file in the sandbox and the Harbor instruction."""
    documents = [c for c in sample.input[0].content if getattr(c, "type", None) == "document"]
    if len(documents) != 1:
        raise ValueError(f"sample {sample.id}: expected one PDF, found {len(documents)}")
    prompt = sample.metadata["prompt"].strip()
    return Sample(
        id=sample.id,
        input=[ChatMessageUser(content=[ContentText(text=INSTRUCTION.format(prompt=prompt))])],
        files={DOCUMENT_PATH: documents[0].document},
        metadata=sample.metadata,
    )


@solver
def read_answer() -> Solver:
    """Use the contents of the answer file as the model output."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            answer = await sandbox().read_file(ANSWER_PATH)
        except FileNotFoundError:
            answer = ""
        # Replace the whole output: assigning "" to `completion` leaves the
        # agent's last chat message in the log.
        state.output = ModelOutput.from_content(state.output.model, answer)
        return state

    return solve


@task
def gdp_pdf_agent(
    hf_repo: str = DEFAULT_HF_REPO,
    hf_revision: str | None = DEFAULT_HF_REVISION,
    limit: int | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    agent: str = "claude_code",
    agent_version: str = "stable",
) -> Task:
    """`agent` is claude_code or codex_cli; pin `agent_version` to Harbor's."""
    samples = load_samples(hf_repo=hf_repo, hf_revision=hf_revision, limit=limit)
    agents = {"claude_code": claude_code, "codex_cli": codex_cli}
    if agent not in agents:
        raise ValueError(f"agent must be one of {sorted(agents)}")
    return Task(
        dataset=MemoryDataset([_agent_sample(s) for s in samples], name="gdp_pdf_agent"),
        solver=[agents[agent](cwd="/workspace", version=agent_version), read_answer()],
        scorer=rubric_scorer(judge_model=judge_model),
        sandbox=("docker", str(SANDBOX_DIR / "compose.yaml")),
        time_limit=1800,
    )
