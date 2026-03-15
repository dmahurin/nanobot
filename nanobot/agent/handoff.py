"""Model handoff routing for top-level agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.events import InboundMessage, OutboundMessage


@dataclass
class HandoffOutcome:
    response: "OutboundMessage | None"
    sent: bool


class HandoffRouter:
    """Routes sessions between top-level agents and executes handoffs."""

    def __init__(self, agents: dict[str, "AgentLoop"], default_name: str = "defaults"):
        self.agents = agents
        self.default_name = default_name
        self.routes: dict[str, str] = {}

    def list_team(self) -> list[dict[str, str]]:
        return [
            {"name": name, "model": agent.model}
            for name, agent in sorted(self.agents.items())
        ]

    def list_agent_names(self, *, exclude: str | None = None) -> list[str]:
        return [
            name for name in sorted(self.agents.keys())
            if name != exclude
        ]

    def route_for(self, session_key: str) -> str:
        return self.routes.get(session_key, self.default_name)

    def set_route(self, session_key: str, agent_name: str) -> None:
        if agent_name in self.agents:
            self.routes[session_key] = agent_name
        else:
            logger.warning("Unknown handoff target '{}'", agent_name)

    async def dispatch(
        self,
        agent_name: str,
        msg: "InboundMessage",
        session_key: str | None = None,
    ) -> "OutboundMessage | None":
        agent = self.agents.get(agent_name)
        if not agent:
            return None
        return await agent._process_message(msg, session_key=session_key or msg.session_key)

    async def handoff(
        self,
        source_name: str,
        target_name: str,
        msg: "InboundMessage",
        session_key: str,
    ) -> HandoffOutcome:
        target = self.agents.get(target_name)
        if not target:
            return HandoffOutcome(response=None, sent=False)

        self.set_route(session_key, target_name)

        source = self.agents.get(source_name)
        if source:
            self._sync_session(source, target, session_key)

        logger.info("Handoff {} -> {}", source_name, target_name)

        response = await target._process_message(msg, session_key=session_key)
        return HandoffOutcome(response=response, sent=response is None)

    def _sync_session(self, source: "AgentLoop", target: "AgentLoop", session_key: str) -> None:
        """Seed target session with source history if target has none yet."""
        source_session = source.sessions.get_or_create(session_key)
        target_session = target.sessions.get_or_create(session_key)

        if target_session.messages:
            return
        if not source_session.messages:
            return

        target_session.messages = list(source_session.messages)
        target_session.last_consolidated = source_session.last_consolidated
        target_session.metadata["handoff_synced_from"] = source.agent_name
        target_session.metadata["handoff_synced_at"] = datetime.now().isoformat()
        target.sessions.save(target_session)
