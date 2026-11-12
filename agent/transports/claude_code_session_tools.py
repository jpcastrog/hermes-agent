"""Which tools the ``claude`` child is started with: its built-ins and the
hermes-tools MCP server's allow rules. Sibling of ``claude_code_session``."""

from __future__ import annotations

from typing import Optional


def hermes_mcp_allow_rules(permission_mode: str) -> list[str]:
    """``--allowedTools`` entries for the hermes-tools MCP server.

    * ``acceptEdits`` (auto): the whole server — every tool is already gated
      by Hermes' own policy (command guards, file policy, hooks).
    * ``default`` (approval-required): every tool except
      :data:`GATED_HERMES_TOOLS`, which fall through to ``can_use_tool`` and
      Hermes' approval prompt.

    The bridged agent-loop tools are named here too. They are dispatched by
    the parent process through ``_invoke_tool`` — the same hooks and the same
    policy as any Hermes turn — so routing them through the CLI's approval
    prompt as well would gate them twice, and in a gateway context (no
    approver) would deny delegation outright.
    """
    from agent.transports.claude_code_session import (
        _MCP_TOOL_PREFIX, GATED_HERMES_TOOLS, HERMES_TOOLS_MCP_SERVER_NAME,
    )

    if permission_mode != "default":
        return [f"mcp__{HERMES_TOOLS_MCP_SERVER_NAME}"]
    from agent.transports.hermes_tool_bridge import BRIDGED_TOOLS
    from agent.transports.hermes_tools_mcp_server import exposed_tools_for_profile

    gated = set(GATED_HERMES_TOOLS)
    names = tuple(exposed_tools_for_profile("claude-code")) + BRIDGED_TOOLS
    return [
        f"{_MCP_TOOL_PREFIX}{name}"
        for name in dict.fromkeys(names)
        if name not in gated
    ]


def native_tool_flags(
    tool_scope: Optional[frozenset], allowed_tools: list[str], disallowed_tools: list[str],
) -> tuple[list[str], list[str]]:
    """``(argv flags, allowed built-ins)`` for the child's own built-in tools.

    The tool-authority boundary (#98533): native OS-level tools are denied in
    EVERY mode, including bypassPermissions, unless the operator set
    claude_code.native_tools. ``--tools`` restricts the built-in set to the
    inert allowlist (anything the CLI adds in a future release is excluded by
    construction); the deny list is belt-and-braces. Deny rules win over allow
    rules and over the permission mode inside the CLI.

    ``tool_scope`` is a request's narrowed tools (X-Hermes-Enabled-Toolsets).
    A narrowed request keeps the boundary even with native_tools on: native
    Bash/WebFetch would walk straight past the narrowing. An empty scope is
    ``--tools ""``: no built-ins at all, inert ones included (verified on the
    real CLI 2.1.153: ``system/init`` reports ``tools: []``).
    """
    from agent.transports.claude_code_session import _INERT_NATIVE_TOOLS, NATIVE_OS_TOOLS

    if tool_scope == frozenset():
        deny = dict.fromkeys(NATIVE_OS_TOOLS + tuple(disallowed_tools))
        return ["--tools", "", "--disallowedTools", ",".join(deny)], []
    flags: list[str] = []
    if disallowed_tools or tool_scope is not None:
        deny = dict.fromkeys(disallowed_tools or NATIVE_OS_TOOLS)
        flags = ["--tools", ",".join(_INERT_NATIVE_TOOLS), "--disallowedTools", ",".join(deny)]
    allowed = [t for t in allowed_tools if t not in disallowed_tools]
    if tool_scope is not None:
        allowed = [t for t in allowed if t in _INERT_NATIVE_TOOLS]
    return flags, allowed
