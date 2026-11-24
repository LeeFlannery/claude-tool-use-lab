"""Fast, key-free checks against the solution notebook's own code.

Executes only the cells tagged "keyfree" (config, data, tools, guard,
eval fixtures, and their check cells), so the code under test is exactly
what learners see. No network, no API key.
"""

import json
from pathlib import Path

import nbformat
import pytest
from IPython.core.inputtransformer2 import TransformerManager

SOLUTION = Path(__file__).resolve().parent.parent / "lab" / "claude-tool-use-lab-solution.ipynb"


@pytest.fixture(scope="module")
def ns():
    nb = nbformat.read(SOLUTION, as_version=4)
    transform = TransformerManager().transform_cell  # turns %pip into get_ipython() calls
    namespace = {"__name__": "lab"}
    for cell in nb.cells:
        if cell.cell_type == "code" and "keyfree" in cell.metadata.get("tags", []):
            exec(compile(transform(cell.source), f"<{cell.id}>", "exec"), namespace)
    return namespace


def test_model_lives_in_one_constant(ns):
    assert ns["MODEL"] == "claude-opus-5-5"
    src = SOLUTION.read_text()
    assert src.count("claude-opus-5-5") == 1, "model ID should appear once, in MODEL"


def test_read_only_tools_are_strict_schemas(ns):
    tools = {t["name"]: t for t in ns["TOOLS"]}
    assert set(tools) == {"list_sodas", "get_soda", "check_stock"}
    for tool in tools.values():
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema.get("required", [])) <= set(schema["properties"])
        assert len(tool["description"]) > 20
        assert tool["name"] in ns["TOOL_FUNCTIONS"]


def test_tools_do_what_their_schemas_say(ns):
    sodas = ns["list_sodas"]()
    assert len(sodas) == 6 and {"soda_id", "name", "flavor"} == set(sodas[0])
    assert ns["get_soda"]("fz-006")["caffeine_mg"] == 140
    assert ns["check_stock"]("fz-002") == {"soda_id": "fz-002", "name": "Fog Machine", "cases": 3, "bottles": 96}


def test_execute_tool_reports_errors_instead_of_raising(ns):
    content, is_error = ns["execute_tool"]("check_stock", {"soda_id": "fz-999"})
    assert is_error and "KeyError" in content
    content, is_error = ns["execute_tool"]("nope", {})
    assert is_error and "Unknown tool" in content


def test_allowlist_is_exactly_the_read_only_tools(ns):
    assert ns["READ_ONLY_ALLOWLIST"] == {t["name"] for t in ns["TOOLS"]}
    assert "update_stock" not in ns["READ_ONLY_ALLOWLIST"]
    assert "update_stock" in ns["TOOL_FUNCTIONS"], "the write tool exists, so the guard has to matter"


def test_guard_denies_writes_and_logs_every_call(ns):
    ns["CALL_LOG"].clear()
    content, is_error = ns["guarded_execute"]("update_stock", {"soda_id": "fz-004", "cases": 99})
    assert is_error and "read-only" in content.lower()
    assert ns["INVENTORY"] == ns["PRISTINE"]

    content, is_error = ns["guarded_execute"]("get_soda", {"soda_id": "fz-001"})
    assert not is_error and json.loads(content)["name"] == "Porch Light"

    content, is_error = ns["guarded_execute"]("drop_tables", {})
    assert is_error

    assert [(e["tool"], e["decision"]) for e in ns["CALL_LOG"]] == [
        ("update_stock", "denied"),
        ("get_soda", "allowed"),
        ("drop_tables", "denied"),
    ]
    assert all(e["at"] for e in ns["CALL_LOG"])


def test_eval_references_match_the_data(ns):
    inv = ns["INVENTORY"]
    refs = " ".join(c["reference"] for c in ns["EVAL_CASES"])
    assert f"{inv['fz-002']['cases']} cases" in refs
    assert f"{inv['fz-006']['caffeine_mg']} mg" in refs
    assert f"{inv['fz-003']['bottles']} bottles" in refs
    assert ns["JUDGE_SCHEMA"]["additionalProperties"] is False


def test_learner_notebook_matches_solution_outside_exercises():
    learner = nbformat.read(SOLUTION.with_name("claude-tool-use-lab.ipynb"), as_version=4)
    solution = nbformat.read(SOLUTION, as_version=4)
    assert len(learner.cells) == len(solution.cells)
    todo_cells = 0
    for lc, sc in zip(learner.cells, solution.cells, strict=True):
        if "exercise" in lc.metadata.get("tags", []):
            assert "TODO" in lc.source and "TODO" not in sc.source
            todo_cells += 1
        elif lc.cell_type == "code":
            assert lc.source == sc.source
    assert todo_cells == 5


def test_no_em_dashes_or_key_literals():
    root = SOLUTION.parent.parent
    for path in [*root.glob("lab/*.ipynb"), root / "README.md", root / "scripts" / "build_notebooks.py"]:
        text = path.read_text()
        assert "\u2014" not in text, f"em dash in {path.name}"
        assert "sk-ant-" not in text, f"key-looking literal in {path.name}"


class ScriptedClient:
    """Stands in for anthropic.Anthropic: replays canned responses, records requests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        return self.responses.pop(0)


def _tool_use(id_, name, input_):
    from types import SimpleNamespace as NS

    return NS(type="tool_use", id=id_, name=name, input=input_)


def _reply(stop_reason, *blocks):
    from types import SimpleNamespace as NS

    return NS(stop_reason=stop_reason, content=[NS(type="thinking", thinking=""), *blocks])


def test_run_agent_loop_with_parallel_calls_and_guard(ns):
    from types import SimpleNamespace as NS

    fake = ScriptedClient([
        _reply("tool_use", _tool_use("t1", "check_stock", {"soda_id": "fz-002"}), _tool_use("t2", "update_stock", {"soda_id": "fz-002", "cases": 50})),
        _reply("end_turn", NS(type="text", text="Fog Machine has 3 cases. I can't change stock.")),
    ])
    ns["client"] = fake
    result = ns["run_agent"]("q", tools=ns["ALL_TOOLS"], executor=ns["guarded_execute"])

    assert result.stop_reason == "end_turn" and result.turns == 2
    assert result.answer == "Fog Machine has 3 cases. I can't change stock."
    assert [n for n, _ in result.tool_calls] == ["check_stock", "update_stock"]
    assert ns["INVENTORY"] == ns["PRISTINE"]

    # The second request carries: user, full assistant reply, ONE user message with both results.
    sent = fake.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[1]["content"][0].type == "thinking", "assistant reply must be appended whole"
    results = sent[2]["content"]
    assert [(r["tool_use_id"], r["is_error"]) for r in results] == [("t1", False), ("t2", True)]


def test_run_agent_stops_at_max_turns(ns):
    loop = [_reply("tool_use", _tool_use(f"t{i}", "list_sodas", {})) for i in range(3)]
    fake = ScriptedClient(loop)
    ns["client"] = fake
    result = ns["run_agent"]("q", max_turns=3)
    assert result.stop_reason == "max_turns" and result.turns == 3
    assert len(fake.requests) == 3


def test_no_alcohol_references():
    import re

    banned = re.compile(r"\b(beers?|kegs?|brew(ery|ing)?|taproom|on tap|ipas?|stouts?|ales?|abv|wine|liquor|cocktails?|booze|pints?|hops?)\b", re.I)
    root = SOLUTION.parent.parent
    for path in [*root.glob("lab/*.ipynb"), root / "README.md", root / "scripts" / "build_notebooks.py"]:
        hits = banned.findall(path.read_text())
        assert not hits, f"alcohol reference in {path.name}: {hits}"
