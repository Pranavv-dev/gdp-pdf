"""GDP.pdf with a CLI coding agent instead of a single completion.

Added for parity with the Harbor adapter. Each sample runs in a Docker sandbox
built from the same Dockerfile as the Harbor task (minus the verifier's judge
client). The PDF is copied to /workspace/document.pdf, the agent gets the
Harbor task's instruction text, and whatever it writes to /workspace/answer.md
becomes the response that `rubric_scorer` grades, unchanged. Claude Code is
launched with the same command, settings and pinned binary as Harbor's
claude-code agent (see `harbor_claude_code`).

    CLAUDE_BIN_DIR=... ANTHROPIC_API_KEY=... ANTHROPIC_BASE_URL=... \
    uv run inspect eval src/gdp_pdf/agent_task.py@gdp_pdf_agent \
      --model anthropic/claude-haiku-4-5 -T agent_model=claude-haiku-4-5 \
      -T agent_version=2.1.311 -T judge_model=...
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentText, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

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


AGENT_LOGS_DIR = "/logs/agent"


@solver
def harbor_claude_code(model: str, version: str, timeout: int = 1800) -> Solver:
    """Run Claude Code exactly as Harbor 0.25.0's `claude-code` agent does.

    Same pinned binary (bind-mounted at /root/.local/bin, as in the Harbor
    run), same environment variables, same setup command and the same
    `claude --verbose --output-format=stream-json
    --permission-mode=bypassPermissions --print` launch, with the instruction
    piped on stdin. Model calls go straight to ANTHROPIC_BASE_URL, as in
    Harbor; nothing is routed through Inspect's model.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        box = sandbox()
        path = 'export PATH="$HOME/.local/bin:$PATH"; '
        got = await box.exec(["bash", "-c", path + "claude --version"])
        if version not in got.stdout:
            raise RuntimeError(f"expected Claude Code {version}, found {got.stdout!r} {got.stderr!r}")

        # ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL reach the container through
        # compose.yaml, because Inspect writes this env dict into its log.
        missing = [k for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL") if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"set {', '.join(missing)} on the host")
        env = {
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
            "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
            "CLAUDE_CONFIG_DIR": f"{AGENT_LOGS_DIR}/sessions",
        }
        setup = (
            "mkdir -p $CLAUDE_CONFIG_DIR/debug $CLAUDE_CONFIG_DIR/projects/-app "
            "$CLAUDE_CONFIG_DIR/shell-snapshots $CLAUDE_CONFIG_DIR/statsig "
            "$CLAUDE_CONFIG_DIR/todos $CLAUDE_CONFIG_DIR/skills && "
            "if [ -d ~/.claude/skills ]; then "
            "cp -r ~/.claude/skills/. $CLAUDE_CONFIG_DIR/skills/ 2>/dev/null || true; "
            "fi"
        )
        done = await box.exec(["bash", "-c", setup], env=env, cwd="/workspace")
        if not done.success:
            raise RuntimeError(f"setup failed: {done.stderr}")

        shell_var = f"claude_code_instruction_{uuid.uuid4().hex}"
        env_var = shell_var.upper()
        instruction = state.input_text
        command = (
            path
            + f'{shell_var}="${env_var}"; unset {env_var}; '
            + f'printf "%s" "${shell_var}" | '
            + "claude --verbose --output-format=stream-json "
            + "--permission-mode=bypassPermissions --print 2>&1 | tee "
            + f"{AGENT_LOGS_DIR}/claude-code.txt"
        )
        try:
            await box.exec(
                ["bash", "-c", command],
                env={**env, env_var: instruction},
                cwd="/workspace",
                timeout=timeout,
            )
        except TimeoutError:
            state.metadata["agent_timeout"] = True

        # Keep Claude Code's own stream log, the same log Harbor keeps, and
        # pull the token usage from its final `result` event.
        try:
            log_text = await box.read_file(f"{AGENT_LOGS_DIR}/claude-code.txt")
        except FileNotFoundError:
            log_text = ""
        log_dir = os.environ.get("CLAUDE_LOG_DIR")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / f"{state.sample_id}.txt").write_text(log_text)
        for line in reversed(log_text.splitlines()):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                state.metadata["claude_code_result"] = {
                    k: event.get(k)
                    for k in ("subtype", "is_error", "num_turns", "total_cost_usd", "usage", "modelUsage")
                }
                break
        return state

    return solve


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
    agent_model: str = "claude-haiku-4-5",
    agent_version: str = "2.1.311",
) -> Task:
    """Claude Code launched the way Harbor's claude-code agent launches it.

    The pinned linux-arm64 binary must be in $CLAUDE_BIN_DIR (mounted by
    compose.yaml), and ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL must be set.
    """
    samples = load_samples(hf_repo=hf_repo, hf_revision=hf_revision, limit=limit)
    return Task(
        dataset=MemoryDataset([_agent_sample(s) for s in samples], name="gdp_pdf_agent"),
        solver=[harbor_claude_code(model=agent_model, version=agent_version), read_answer()],
        scorer=rubric_scorer(judge_model=judge_model),
        sandbox=("docker", str(SANDBOX_DIR / "compose.yaml")),
        time_limit=1800,
    )
