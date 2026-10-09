import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent import MockLLM, QualityAgent
from agent.tools import calculate_capability, check_tolerance, pareto_defects, run_tool

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cpk_known_value():
    r = calculate_capability([9, 10, 11, 10, 10, 10], 6, 14)
    assert r["cpk"] > 1.5


def test_tolerance_counts():
    r = check_tolerance([1, 2, 3, 10], 0, 5)
    assert r["out_of_tolerance"] == 1 and r["reject_rate_pct"] == 25.0


def test_pareto_order():
    r = pareto_defects(["a", "b", "a", "a"])
    assert r["ranking"][0]["defect"] == "a" and r["ranking"][-1]["cumulative_pct"] == 100.0


def test_tool_errors_are_observations():
    assert "error" in run_tool("nope", {})
    assert "error" in run_tool("check_tolerance", {"bad": 1})


def test_agent_rejects_bad_batch():
    task = json.loads((ROOT / "sample_data/batch_bad.json").read_text())
    out = QualityAgent(MockLLM(), verbose=False).run(task)
    assert out["result"]["verdict"].startswith("REJECT")


def test_agent_accepts_good_batch():
    task = json.loads((ROOT / "sample_data/batch_ok.json").read_text())
    out = QualityAgent(MockLLM(), verbose=False).run(task)
    assert out["result"]["verdict"] == "ACCEPT"
