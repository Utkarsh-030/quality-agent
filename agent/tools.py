"""Tools the agent can call. Pure Python, standard library only."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any


def calculate_capability(measurements: list[float], lsl: float, usl: float) -> dict[str, Any]:
    """Compute mean, std dev, Cp and Cpk for a set of measurements."""
    if len(measurements) < 2:
        raise ValueError("Need at least 2 measurements")
    if usl <= lsl:
        raise ValueError("usl must be greater than lsl")
    mean = statistics.mean(measurements)
    sd = statistics.stdev(measurements)
    if sd == 0:
        return {"n": len(measurements), "mean": mean, "std_dev": 0.0, "cp": None, "cpk": None,
                "note": "Zero variation; capability indices undefined"}
    cp = (usl - lsl) / (6 * sd)
    cpk = min(usl - mean, mean - lsl) / (3 * sd)
    return {"n": len(measurements), "mean": round(mean, 4), "std_dev": round(sd, 4),
            "cp": round(cp, 3), "cpk": round(cpk, 3)}


def check_tolerance(measurements: list[float], lsl: float, usl: float) -> dict[str, Any]:
    """Count out-of-tolerance parts and return their indices."""
    out = [(i, v) for i, v in enumerate(measurements) if v < lsl or v > usl]
    n = len(measurements)
    return {"n": n, "out_of_tolerance": len(out),
            "reject_rate_pct": round(100 * len(out) / n, 2) if n else 0.0,
            "violations": [{"index": i, "value": v} for i, v in out]}


def detect_trend(measurements: list[float]) -> dict[str, Any]:
    """Flag drift: 7+ consecutive points rising/falling, or 7+ on one side of the mean."""
    if len(measurements) < 7:
        return {"drift_detected": False, "reason": "fewer than 7 points"}
    mean = statistics.mean(measurements)
    run_dir, best_run, direction = 1, 1, 0
    for a, b in zip(measurements, measurements[1:]):
        d = 1 if b > a else -1 if b < a else 0
        if d != 0 and d == direction:
            run_dir += 1
        else:
            run_dir = 2 if d != 0 else 1
        direction = d
        best_run = max(best_run, run_dir)
    side_run = best_run_side = 0
    last_side = 0
    for v in measurements:
        s = 1 if v > mean else -1 if v < mean else 0
        side_run = side_run + 1 if (s == last_side and s != 0) else (1 if s != 0 else 0)
        last_side = s
        best_run_side = max(best_run_side, side_run)
    drift = best_run >= 7 or best_run_side >= 7
    return {"drift_detected": drift, "longest_monotonic_run": best_run,
            "longest_same_side_run": best_run_side}


def pareto_defects(defects: list[str]) -> dict[str, Any]:
    """Rank defect types by frequency with cumulative percentage."""
    if not defects:
        return {"total": 0, "ranking": []}
    counts = Counter(defects)
    total = sum(counts.values())
    cum, ranking = 0, []
    for name, c in counts.most_common():
        cum += c
        ranking.append({"defect": name, "count": c, "pct": round(100 * c / total, 1),
                        "cumulative_pct": round(100 * cum / total, 1)})
    return {"total": total, "ranking": ranking}


_CAUSE_KB = {
    "leak": ["Weld porosity or incomplete fusion", "Damaged/missing seal or gasket", "Over/under-torqued fasteners"],
    "dimension": ["Tool wear or fixture looseness", "Thermal expansion / setup drift", "Operator measurement error"],
    "weld": ["Incorrect current/travel speed", "Contaminated base metal", "Poor fit-up gap"],
    "scratch": ["Handling damage between stations", "Missing protective packaging", "Debris on fixture"],
    "crack": ["Excess residual stress", "Material defect", "Over-forming / springback compensation error"],
}


def suggest_root_causes(defect: str) -> dict[str, Any]:
    """Return candidate root causes plus 5-Why starter questions for a defect."""
    key = next((k for k in _CAUSE_KB if k in defect.lower()), None)
    causes = _CAUSE_KB.get(key, ["Man / Machine / Material / Method / Measurement / Environment review needed"])
    return {"defect": defect, "candidate_causes": causes,
            "five_why_start": f"Why did '{defect}' occur on this batch, and why was it not caught earlier?"}


TOOLS = {
    "calculate_capability": calculate_capability,
    "check_tolerance": check_tolerance,
    "detect_trend": detect_trend,
    "pareto_defects": pareto_defects,
    "suggest_root_causes": suggest_root_causes,
}


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool safely; errors are returned as observations so the agent can recover."""
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(TOOLS)}"}
    try:
        return TOOLS[name](**args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # noqa: BLE001 - surface to the agent as an observation
        return {"error": str(e)}
