"""A request's narrowed tools on the claude_code runtime (X-Hermes-Enabled-Toolsets).

API-key providers honour the narrowing through the agent's own tool list. Under this runtime the
tools live in the hermes-tools MCP server, which builds its list from the whole registry in another
process, so the narrowed names are handed to the child at spawn (``ClaudeCodeSession(tool_scope=)``)
and a warm child built for another scope is never reused. Sibling of ``claude_code_runtime``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def hermes_tool_scope(agent) -> Optional[frozenset]:
    """The tool names ``agent``'s child may see, or ``None`` when its request did not narrow them.

    Only an agent api_server marked with ``_hermes_restricted_toolsets`` is narrowed; every other
    agent keeps today's surface exactly, whatever its ``valid_tool_names`` say. An empty result
    means no tools at all, never "no restriction".
    """
    if getattr(agent, "_hermes_restricted_toolsets", None) is None:
        return None
    names = set(getattr(agent, "valid_tool_names", None) or ())
    if "tool_call" in names:
        # Deferred tools are still the agent's to run via tool_call (same authority as
        # claude_code_runtime.bridged_tools_for).
        from agent.tool_executor import _tool_search_scoped_names
        names |= set(_tool_search_scoped_names(agent))
    return frozenset(names)


def scope_signature(agent) -> Optional[tuple]:
    """The scope as a hashable part of a spare's spawn signature."""
    scope = hermes_tool_scope(agent)
    return None if scope is None else tuple(sorted(scope))


def rebuild_if_out_of_scope(agent, entry: Any, registry_key: Optional[str]) -> bool:
    """Replace ``entry``'s warm child when it was built for another tool scope; True if replaced.

    The scope is baked in at spawn (mcp-config, ``--tools``). A no-tools child must never serve a
    full request, nor a full child a no-tools one. Called holding the entry's turn lock; on a
    failed rebuild the entry is dropped and the lock released before the error propagates. The
    CLI transcript is resumed by the new child as usual. Narrowed requests should use their own
    session key: one key shared by narrowed and full requests rebuilds the child on every switch.
    """
    from agent import claude_code_runtime as rt

    old = entry.session
    if getattr(old, "tool_scope", None) == hermes_tool_scope(agent):
        return False
    logger.info("claude-code: tool scope changed; rebuilding session %s", (registry_key or "-")[:12])
    try:
        old.close()
        entry.session = rt._build_session(agent)
    except BaseException:
        rt.evict_session(registry_key)
        entry.turn_lock.release()
        raise
    rt._carry_turn_ledger(old, entry.session)
    return True
