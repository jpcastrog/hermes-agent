"""A private chat (api_server ``X-Hermes-Ephemeral``) on the claude_code runtime.

The child must keep no transcript (``--no-session-persistence``), never be resumed or
registered in the warm-process pool, be closed when the turn ends, and get the earlier turns
from the request body in its own input, because nothing stored can give them to it. Driven
end to end against the fake ``claude`` CLI the runtime tests use; each check has an ordinary
control showing the opposite.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent import claude_code_runtime as rt
from agent.transports.claude_code_session import ClaudeCodeSession
from tests.agent.test_claude_code_runtime import _agent, _env  # noqa: F401  (autouse fixture)

EARLIER = "remember the pharmacy on Calle Mayor"
NOW = "and what time does it open?"


def _messages():
    return [
        {"role": "user", "content": EARLIER},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": NOW},
    ]


def _run(agent, tmp_path: Path, monkeypatch):
    record, inbox = tmp_path / "record.json", tmp_path / "inbox.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_RECORD", str(record))
    monkeypatch.setenv("FAKE_CLAUDE_INBOX", str(inbox))
    result = rt.run_claude_code_turn(
        agent, user_message=NOW, original_user_message=NOW,
        messages=_messages(), effective_task_id="t",
    )
    argv = json.loads(record.read_text())["argv"]
    sent = [json.loads(line) for line in inbox.read_text().splitlines()]
    return result, argv, json.dumps(sent[-1])


def test_private_turn_runs_a_one_off_child_with_no_transcript(tmp_path, monkeypatch):
    agent = _agent("ephemeral-abc")
    agent._ephemeral_turn = True
    result, argv, last_input = _run(agent, tmp_path, monkeypatch)

    assert result["completed"] is True
    assert "--no-session-persistence" in argv
    assert "--resume" not in argv and "--session-id" not in argv
    assert rt.registered_session_count() == 0
    assert agent._claude_code_session._closed is True
    # The history reached the child in the turn itself.
    assert EARLIER in last_input and NOW in last_input


def test_control_ordinary_turn_is_registered_and_persisted(tmp_path, monkeypatch):
    agent = _agent("ordinary-abc")
    result, argv, last_input = _run(agent, tmp_path, monkeypatch)

    assert result["completed"] is True
    assert "--no-session-persistence" not in argv
    assert "--session-id" in argv
    assert rt.registered_session_count() == 1
    assert agent._claude_code_session._closed is False
    assert EARLIER not in last_input  # an ordinary turn's history lives in the transcript


def test_no_persistence_overrides_resume(tmp_path):
    session = ClaudeCodeSession(
        config_dir=str(tmp_path), session_id="s", session_key="k", resume=True, persist=False,
        warmup=False)
    argv = session._build_command(resume=True)
    assert "--no-session-persistence" in argv
    assert "--resume" not in argv and "--session-id" not in argv
    session.claim("someone")
    assert not (tmp_path / "session-map.json").exists()
    assert list(tmp_path.glob("*map*")) == []


def test_history_goes_in_front_of_the_new_message():
    text = rt._ephemeral_input(_messages(), NOW)
    assert text.index(EARLIER) < text.index("Noted.") < text.rindex(NOW)
    assert rt._ephemeral_input([{"role": "user", "content": NOW}], NOW) == NOW
