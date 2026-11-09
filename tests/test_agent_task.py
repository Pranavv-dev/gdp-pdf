"""The agent task reads its answer from the sandbox (needs Docker)."""

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, scorer, mean
from inspect_ai.solver import solver
from inspect_ai.util import sandbox

from gdp_pdf.agent_task import ANSWER_PATH, DOCUMENT_PATH, SANDBOX_DIR, read_answer


@solver
def write_answer():
    async def solve(state, generate):
        # The document must already be in place before the agent starts.
        assert (await sandbox().exec(["test", "-s", DOCUMENT_PATH])).success
        await sandbox().write_file(ANSWER_PATH, "the answer")
        return state

    return solve


@scorer(metrics=[mean()])
def seen():
    async def score(state, target):
        return Score(value=1.0 if state.output.completion == "the answer" else 0.0)

    return score


def test_answer_file_becomes_output(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    task = Task(
        dataset=[Sample(input="q", files={DOCUMENT_PATH: str(pdf)})],
        solver=[write_answer(), read_answer()],
        scorer=seen(),
        sandbox=("docker", str(SANDBOX_DIR / "compose.yaml")),
    )
    [log] = eval(task, model="mockllm/model", display="none", log_dir=str(tmp_path))
    assert log.status == "success"
    assert log.results.scores[0].metrics["mean"].value == 1.0
