"""LLM backends. The agent only needs `chat(messages) -> str` returning JSON text.

Backends:
  - GeminiLLM : Google Gemini via REST (key read from GEMINI_API_KEY env var)
  - MockLLM   : deterministic scripted policy, so the project runs and tests offline
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class GeminiLLM:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Put it in your environment, never in code.")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    def chat(self, messages: list[dict]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Gemini API error {e.code}: {e.read().decode()[:300]}") from None
        return data["candidates"][0]["content"]["parts"][0]["text"]


class MockLLM:
    """Scripted policy: runs the standard inspection workflow, then writes a verdict.

    It reads tool observations from the conversation, so the final answer reflects real tool output.
    """

    STEPS = ["check_tolerance", "calculate_capability", "detect_trend", "pareto_defects", "suggest_root_causes"]

    def chat(self, messages: list[dict]) -> str:
        task = json.loads(next(m["content"] for m in messages if m["role"] == "user"))
        obs = [json.loads(m["content"][len("OBSERVATION: "):]) for m in messages
               if m["role"] == "user" and m["content"].startswith("OBSERVATION: ")]
        step = len(obs)
        m, lsl, usl = task["measurements"], task["lsl"], task["usl"]
        if step == 0:
            return json.dumps({"thought": "Check tolerance first.", "action": "check_tolerance",
                               "args": {"measurements": m, "lsl": lsl, "usl": usl}})
        if step == 1:
            return json.dumps({"thought": "Quantify process capability.", "action": "calculate_capability",
                               "args": {"measurements": m, "lsl": lsl, "usl": usl}})
        if step == 2:
            return json.dumps({"thought": "Look for drift.", "action": "detect_trend", "args": {"measurements": m}})
        if step == 3 and task.get("defects"):
            return json.dumps({"thought": "Rank defects.", "action": "pareto_defects",
                               "args": {"defects": task["defects"]}})
        if step in (3, 4) and task.get("defects"):
            top = obs[3]["ranking"][0]["defect"]
            return json.dumps({"thought": "Root causes for the top defect.", "action": "suggest_root_causes",
                               "args": {"defect": top}})
        tol, cap, trend = obs[0], obs[1], obs[2]
        cpk = cap.get("cpk")
        accept = tol["out_of_tolerance"] == 0 and cpk is not None and cpk >= 1.33 and not trend["drift_detected"]
        verdict = "ACCEPT" if accept else "REJECT / HOLD FOR REVIEW"
        causes = obs[-1].get("candidate_causes", []) if obs and "candidate_causes" in obs[-1] else []
        return json.dumps({"final_answer": {
            "verdict": verdict,
            "reasons": [f"{tol['out_of_tolerance']} of {tol['n']} parts out of tolerance",
                        f"Cpk = {cpk} (target >= 1.33)", f"Drift detected: {trend['drift_detected']}"],
            "recommended_actions": causes or ["No defect data supplied; continue routine monitoring"]}})
