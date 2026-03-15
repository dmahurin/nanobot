"""Handoff tool for transferring control to another top-level agent."""

from typing import Any

from nanobot.agent.tools.base import Tool


class TransferTool(Tool):
    """Tool definition for model handoff."""

    def __init__(self, target: str):
        self._target = target

    @property
    def name(self) -> str:
        return f"transfer_to_{self._target}"

    @property
    def description(self) -> str:
        return (
            f"Hand off this conversation to the {self._target} agent. "
            "Use when another agent is better suited. This is invisible to the user."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **_kwargs: Any) -> str:
        # Actual handoff is handled by the agent loop; this is a schema stub.
        return "Handoff requested."
