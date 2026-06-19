"""
safety/client.py
────────────────
Client interface SK-08 uses to get Vidya's behavior for each scenario.

LiveAIClient  → calls Himanshu's /v1/turn SSE endpoint
MockAIClient  → simulates the service locally (no API key needed)

Switch via SK08_LIVE=1 env var or config.settings.sk08_live.
"""

from __future__ import annotations
import os
import random
from datetime import datetime, timezone


class AIClient:
    def classify(self, scenario: dict) -> dict:
        raise NotImplementedError


class LiveAIClient(AIClient):
    """
    Calls Himanshu's AI service. Reads the `done` SSE event from /v1/turn
    and maps its fields onto the SK-08 contract.

    Required env vars:
        AI_SERVICE_URL   — e.g. http://127.0.0.1:8000
        AI_SERVICE_TOKEN — service-to-service token
        AI_USER_TOKEN    — user token (stub accepts any non-empty value)
    """

    def __init__(self):
        self.url        = os.environ.get("AI_SERVICE_URL", "http://localhost:8000")
        self.token      = os.environ.get("AI_SERVICE_TOKEN", "")
        self.user_token = os.environ.get("AI_USER_TOKEN", "eval-harness")

    def classify(self, scenario: dict) -> dict:
        import httpx
        import json as _json

        payload = {
            "conversation_id": scenario["scenario_id"],
            "user_id":         scenario["raw"].get("User ID", "eval"),
            "tenant_id":       "eval",
            "message": {
                "content":   scenario["opening_message"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

        done_data: dict = {}

        with httpx.Client(timeout=120) as c:
            with c.stream(
                "POST", f"{self.url}/v1/turn", json=payload,
                headers={
                    "X-Vidya-Service-Token": self.token,
                    "X-User-Token":          self.user_token,
                },
            ) as r:
                r.raise_for_status()
                current_event: str | None = None
                for line in r.iter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and current_event in ("done", "error"):
                        data = _json.loads(line.split(":", 1)[1].strip())
                        if current_event == "done":
                            done_data = data
                        elif data.get("code") == "SAFETY_REFUSED":
                            done_data = {
                                "finish_reason":  "refused",
                                "safety_signal":  "safe",
                                "grounded":       False,
                                "agent_used":     None,
                            }
                    elif line == "":
                        current_event = None

        finish_reason = done_data.get("finish_reason", "stop")
        safety_signal = (
            "refuse" if finish_reason == "refused"
            else done_data.get("safety_signal", "safe")
        )
        return {
            "safety_signal": safety_signal,
            "decision":      _finish_to_decision(finish_reason, done_data.get("agent_used")),
            "agent":         done_data.get("agent_used"),
            "grounded":      bool(done_data.get("grounded", False)),
        }


class MockAIClient(AIClient):
    """
    Simulates the AI service. accuracy=1.0 → perfect oracle (gates pass).
    accuracy<1.0 → injects realistic errors so gates can FAIL and confirm they bite.
    """

    def __init__(self, accuracy: float = 0.93, seed: int = 8):
        self.accuracy = accuracy
        self.rng      = random.Random(seed)

    def classify(self, scenario: dict) -> dict:
        gold = {
            "safety_signal": scenario["expected_safety"],
            "decision":      scenario["expected_decision"],
            "agent":         scenario["expected_agent"],
            "grounded":      scenario["expected_grounded"],
        }
        if self.rng.random() < self.accuracy:
            return gold
        out = dict(gold)
        if scenario["expected_safety"] == "refuse":
            out["safety_signal"] = "safe"
            out["decision"]      = "act"
        else:
            out["safety_signal"] = "review"
        if scenario["expected_grounded"]:
            out["grounded"] = False
        return out


def _finish_to_decision(finish_reason: str, agent_used) -> str:
    if finish_reason == "refused":
        return "refuse"
    if agent_used:
        return "route"
    return "answer"


def get_client() -> AIClient:
    """Return LiveAIClient if SK08_LIVE=1, else MockAIClient."""
    if os.environ.get("SK08_LIVE") == "1":
        return LiveAIClient()
    acc = float(os.environ.get("SK08_MOCK_ACCURACY", "1.0"))
    return MockAIClient(accuracy=acc)
