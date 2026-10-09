"""Agent loop: Reason -> Act (tool call) -> Observe -> repeat until a final answer."""
from __future__ import annotations

import json
from typing import Any

from .tools import TOOLS, run_tool

SYSTEM_PROMPT = """You are a Manufacturing Quality Inspection Agent.
You receive a batch's inspection data as JSON and must decide ACCEPT or REJECT, with reasons and corrective actions.

Work step by step. On every turn reply with ONE JSON object and nothing else:
  To use a tool:  {"thought": "...", "action": "<tool_name>", "args": {...}}
  To finish:      {"final_answer": {"verdict": "ACCEPT|REJECT", "reasons": [...], "recommended_actions": [...]}}

Available tools:
{tool_docs}

Rules:
- Base every claim on tool observations, never guess numbers.
- Process is capable if Cpk >= 1.33. Any out-of-tolerance part or detected drift needs a flagged reason.
- If a tool returns an error, fix the arguments and retry or choose another tool.
- Finish within the step limit."""


def _tool_docs() -> str:
    return "\n".join(f"- {n}: {(f.__doc__ or '').strip()}" for n, f in TOOLS.items())


class QualityAgent:
    def __init__(self, llm, max_steps: int = 8, verbose: bool = True):
        self.llm, self.max_steps, self.verbose = llm, max_steps, verbose
        self.trace: list[dict[str, Any]] = []

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.replace("{tool_docs}", _tool_docs())},
            {"role": "user", "content": json.dumps(task)},
        ]
        for step in range(1, self.max_steps + 1):
            raw = self.llm.chat(messages)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": 'OBSERVATION: {"error": "Reply must be valid JSON"}'}]
                continue
            messages.append({"role": "assistant", "content": raw})
            if "final_answer" in msg:
                self._log(step, "FINAL", msg["final_answer"])
                return {"result": msg["final_answer"], "steps": step, "trace": self.trace}
            action, args = msg.get("action"), msg.get("args", {})
            self._log(step, f"THINK: {msg.get('thought', '')}", None)
            observation = run_tool(action, args)
            self._log(step, f"ACT: {action}", observation)
            messages.append({"role": "user", "content": "OBSERVATION: " + json.dumps(observation)})
        return {"result": {"verdict": "INCONCLUSIVE", "reasons": ["Step limit reached"],
                           "recommended_actions": ["Escalate to a human inspector"]},
                "steps": self.max_steps, "trace": self.trace}

    def _log(self, step: int, label: str, payload: Any) -> None:
        self.trace.append({"step": step, "label": label, "payload": payload})
        if self.verbose:
            print(f"[step {step}] {label}")
            if payload is not None:
                print("   ", json.dumps(payload)[:300])
