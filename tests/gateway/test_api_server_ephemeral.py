"""Private chats: ``X-Hermes-Ephemeral: 1`` on POST /v1/chat/completions.

A private chat from the Mac app sends the whole conversation in ``messages`` and must leave
nothing about it on disk: no state.db session or message rows, no ``request_dump_*`` file,
no response_store row, no conversation text in any log record. Everything here is real except
the model: a real ``APIServerAdapter`` and ``AIAgent`` with the state DB in the test's own
HERMES_HOME (the conftest sandbox), answered by the recording loopback ``FakeLLMServer``.
Each private-chat assertion has a control without the header that shows the same turn DOES
leave the trace, so the test cannot pass because nothing was ever written.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text, ToolCall, write_hermes_home
from tests.gateway.test_api_server import _create_app, _make_adapter

SECRET = "my sister's diagnosis is lupus"
EARLIER = "remember the pharmacy on Calle Mayor"
REPLY = "I am sorry to hear that"
KEY = "sk-test-gateway"
AUTH = {"Authorization": f"Bearer {KEY}", "X-Hermes-Session-Key": "moe"}
PRIVATE = {**AUTH, "X-Hermes-Ephemeral": "1"}


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _body(stream: bool = False) -> dict:
    return {
        "model": "fake-model",
        "stream": stream,
        "messages": [
            {"role": "system", "content": "You are Moe."},
            {"role": "user", "content": EARLIER},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": SECRET},
        ],
    }


def _rows(sql: str, db: Path) -> list:
    if not db.exists():
        return []
    con = sqlite3.connect(str(db))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _sessions() -> list:
    return _rows("SELECT id FROM sessions", _home() / "state.db")


def _messages() -> list:
    return _rows("SELECT session_id, content FROM messages", _home() / "state.db")


def _dumps() -> list:
    return sorted(p.name for p in _home().rglob("request_dump_*"))


@pytest.fixture
def fake_model(monkeypatch):
    def _start(script):
        srv = FakeLLMServer(script)
        srv.__enter__()
        write_hermes_home(_home(), srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-e2e")
        started.append(srv)
        return srv

    started: list = []
    yield _start
    for srv in started:
        srv.__exit__(None, None, None)


async def _post(body: dict, headers: dict):
    adapter = _make_adapter(KEY)
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.post("/v1/chat/completions", json=body, headers=headers)
        text = await resp.text()
        return resp.status, dict(resp.headers), text


# ── state.db ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_chat_stores_no_session_or_messages(fake_model):
    srv = fake_model([Text(REPLY)])
    status, headers, text = await _post(_body(), PRIVATE)

    assert status == 200, text
    assert json.loads(text)["choices"][0]["message"]["content"] == REPLY
    assert headers.get("X-Hermes-Ephemeral-Applied") == "1"
    # Nothing to continue: the throwaway id is not handed back.
    assert "X-Hermes-Session-Id" not in headers
    assert _sessions() == []
    assert _messages() == []
    # The child got the whole history from the body.
    sent = srv.main_requests()[-1]["messages"]
    assert any(EARLIER in json.dumps(m) for m in sent)


@pytest.mark.asyncio
async def test_control_ordinary_chat_still_persists(fake_model):
    fake_model([Text(REPLY)])
    status, headers, text = await _post(_body(), AUTH)

    assert status == 200, text
    assert "X-Hermes-Ephemeral-Applied" not in headers
    assert headers.get("X-Hermes-Session-Id")
    assert _sessions() != []
    assert any(SECRET in str(content) for _sid, content in _messages())


@pytest.mark.asyncio
async def test_private_streaming_chat_stores_nothing_and_echoes_header(fake_model):
    fake_model([Text(REPLY)])
    status, headers, text = await _post(_body(stream=True), PRIVATE)

    assert status == 200, text
    assert headers.get("X-Hermes-Ephemeral-Applied") == "1"
    assert "X-Hermes-Session-Id" not in headers
    assert "data: [DONE]" in text
    assert _sessions() == []
    assert _messages() == []


# ── request dumps ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_chat_writes_no_request_dump(fake_model):
    # A non-retryable 400 is what makes the agent dump the full request body to sessions/.
    fake_model([Error(400, "scripted bad request")])
    await _post(_body(), PRIVATE)
    assert _dumps() == []


@pytest.mark.asyncio
async def test_control_ordinary_failed_chat_writes_a_request_dump(fake_model):
    fake_model([Error(400, "scripted bad request")])
    await _post(_body(), AUTH)
    assert _dumps() != []


# ── response store ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_chat_writes_no_response_store_row(fake_model):
    fake_model([Text(REPLY)])
    status, _headers, text = await _post(_body(), PRIVATE)
    assert status == 200, text
    assert _rows("SELECT response_id FROM responses", _home() / "response_store.db") == []


# ── logs ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_chat_logs_no_conversation_text(fake_model, caplog, capfd):
    # The provider's error echoes the person's words: the worst case for logs and the console
    # (the app keeps the gateway's stdout and stderr in a log file).
    fake_model([Error(400, f"provider echoed: {SECRET}")])
    with caplog.at_level(logging.DEBUG):
        await _post(_body(), PRIVATE)
    out, err = capfd.readouterr()
    assert SECRET not in out and SECRET not in err
    logged = "\n".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    assert SECRET not in logged
    assert EARLIER not in logged
    assert "private chat turn" in logged  # the one content-free line saying it ran


@pytest.mark.asyncio
async def test_control_ordinary_chat_logs_the_text(fake_model, caplog, capfd):
    fake_model([Error(400, f"provider echoed: {SECRET}")])
    with caplog.at_level(logging.DEBUG):
        await _post(_body(), AUTH)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET in logged
    out, err = capfd.readouterr()
    assert SECRET in out + err


def test_log_records_inside_a_private_turn_carry_no_text():
    from hermes_logging import EPHEMERAL_LOG_TEXT, reset_ephemeral_turn, set_ephemeral_turn

    log = logging.getLogger("test.ephemeral")
    factory = logging.getLogRecordFactory()
    token = set_ephemeral_turn(True)
    try:
        rec = factory(log.name, logging.INFO, __file__, 1, "said %s", (SECRET,), None)
    finally:
        reset_ephemeral_turn(token)
    assert rec.getMessage() == EPHEMERAL_LOG_TEXT
    after = factory(log.name, logging.INFO, __file__, 1, "said %s", (SECRET,), None)
    assert SECRET in after.getMessage()


def test_a_process_started_for_a_private_turn_logs_no_text(monkeypatch):
    from hermes_logging import EPHEMERAL_LOG_TEXT, EPHEMERAL_TURN_ENV

    monkeypatch.setenv(EPHEMERAL_TURN_ENV, "1")
    rec = logging.getLogRecordFactory()("t", logging.INFO, __file__, 1, "said %s", (SECRET,), None)
    assert rec.getMessage() == EPHEMERAL_LOG_TEXT


# ── the agent it builds ──────────────────────────────────────────────────────


def test_private_agent_has_no_memory_review_or_persistence(fake_model):
    fake_model([])
    adapter = _make_adapter(KEY)
    agent = adapter._create_agent(session_id="ephemeral-x", ephemeral=True)
    try:
        assert agent._persist_disabled is True
        assert agent._ephemeral_turn is True
        assert agent._session_db is None
        assert agent._memory_manager is None
        assert agent._memory_store is None
        assert "memory" not in agent.valid_tool_names
        assert agent.skip_background_review is True
        assert agent._checkpoint_mgr.enabled is False
    finally:
        agent.close()

    ordinary = adapter._create_agent(session_id="ordinary-x")
    try:
        assert ordinary._session_db is not None
        assert getattr(ordinary, "_ephemeral_turn", False) is False
    finally:
        ordinary.close()


# ── the contract around the header ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_private_chat_cannot_continue_a_stored_session(fake_model):
    fake_model([Text(REPLY)])
    status, _headers, text = await _post(_body(), {**PRIVATE, "X-Hermes-Session-Id": "s1"})
    assert status == 400, text
    assert _sessions() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "false", "yes", ""])
async def test_other_header_values_are_off(fake_model, value):
    fake_model([Text(REPLY)])
    status, headers, text = await _post(_body(), {**AUTH, "X-Hermes-Ephemeral": value})
    assert status == 200, text
    assert "X-Hermes-Ephemeral-Applied" not in headers
    assert _sessions() != []


@pytest.mark.asyncio
async def test_capabilities_advertise_the_header():
    adapter = _make_adapter(KEY)
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        resp = await cli.get("/v1/capabilities", headers=AUTH)
        data = await resp.json()
    assert data["features"]["ephemeral_header"] is True


# ── grounding files ──────────────────────────────────────────────────────────


def test_private_turn_end_removes_its_grounding_files(tmp_path):
    from agent import recipient_grounding as rg
    from gateway.platforms.api_server import _forget_ephemeral_turn

    rg._update("ephemeral-k", lambda d: d.__setitem__("words", [{"text": SECRET}]))
    assert rg.ledger_path("ephemeral-k").exists()

    class _Agent:
        _grounding_turn_keys = ["ephemeral-k"]
        session_id = "ephemeral-k"
        _memory_manager = None

    _forget_ephemeral_turn(_Agent())
    assert not rg.ledger_path("ephemeral-k").exists()
    assert not Path(str(rg.ledger_path("ephemeral-k")) + ".lock").exists()


# ── helpers a private turn starts ────────────────────────────────────────────


@pytest.mark.parametrize("private", [True, False], ids=["private", "control-ordinary"])
def test_a_private_turns_helper_is_private_too(fake_model, monkeypatch, private):
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc

    fake_model([])
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    adapter = _make_adapter(KEY)
    parent = adapter._create_agent(session_id="ephemeral-p" if private else "p", ephemeral=private)
    child = dt._build_child_agent(task_index=0, goal="goal", context=None, toolsets=["file"], model=None,
                                  max_iterations=4, task_count=1, parent_agent=parent)
    try:
        assert (getattr(child, "_ephemeral_turn", False) is True) is private
        assert (getattr(child, "_persist_disabled", False) is True) is private
        assert (child in (getattr(parent, "_ephemeral_children", None) or [])) is private
    finally:
        child.close()
        parent.close()


# ── files tools write during a private turn ──────────────────────────────────


def _tool_writes():
    """Every tool writer a private turn redirects, each given the person's words."""
    from tools import delegate_tool_results, tool_result_storage, vision_tools, web_tools_truncate
    from tools.delegation_live_log import live_transcript_root

    paths = [
        tool_result_storage._write_to_spillover(SECRET * 1000, "toolu_1.txt"),
        delegate_tool_results._spill_summary_to_file(0, SECRET),
        web_tools_truncate._store_full_text("https://example.invalid/page", SECRET),
    ]
    live = live_transcript_root() / "child.log"
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_text(SECRET)
    vision = vision_tools._secure_cache_dir("cache/vision", "temp_vision_images") / "img.img"
    vision.write_text(SECRET)
    return [Path(p) for p in paths if p] + [live, vision]


def _files_holding_secret() -> list:
    return [p for p in _home().rglob("*") if p.is_file() and SECRET in p.read_text(errors="ignore")]


def test_private_turn_tool_files_are_gone_when_it_ends():
    from hermes_logging import reset_ephemeral_turn, set_ephemeral_turn

    token = set_ephemeral_turn(True)
    try:
        written = _tool_writes()
        assert len(written) == 5 and all(p.exists() for p in written)
        # During the turn they sit in the turn's scratch folder, nowhere else.
        assert all("cache/private/" in str(p) for p in written), written
    finally:
        reset_ephemeral_turn(token)
    assert not any(p.exists() for p in written)
    assert _files_holding_secret() == []


def test_control_ordinary_tool_files_stay():
    written = _tool_writes()
    assert len(written) == 5
    assert not any("cache/private/" in str(p) for p in written)
    assert len(_files_holding_secret()) == 5


@pytest.mark.asyncio
async def test_private_chat_tool_output_files_are_gone_after_the_turn(fake_model, tmp_path):
    big = tmp_path / "big.txt"
    big.write_text((SECRET + "\n") * 20000)
    fake_model([ToolCall("terminal", {"command": f"cat {big}"}), Text(REPLY)])
    status, _headers, text = await _post(_body(), PRIVATE)
    assert status == 200, text
    spill = _home() / "cache"
    assert [p for p in spill.rglob("*") if p.is_file() and SECRET in p.read_text(errors="ignore")] == []


@pytest.mark.asyncio
async def test_control_ordinary_chat_tool_output_files_stay(fake_model, tmp_path):
    big = tmp_path / "big.txt"
    big.write_text((SECRET + "\n") * 20000)
    fake_model([ToolCall("terminal", {"command": f"cat {big}"}), Text(REPLY)])
    status, _headers, text = await _post(_body(), AUTH)
    assert status == 200, text
    spill = _home() / "cache"
    assert [p for p in spill.rglob("*") if p.is_file() and SECRET in p.read_text(errors="ignore")] != []


def test_private_turn_moa_traces_go_to_scratch(monkeypatch):
    import hermes_cli.config as cfg
    from agent.moa_trace import _traces_enabled_and_dir
    from hermes_logging import reset_ephemeral_turn, set_ephemeral_turn

    monkeypatch.setattr(cfg, "load_config", lambda: {"moa": {"save_traces": True}})
    assert _traces_enabled_and_dir() == _home() / "moa-traces"  # control
    token = set_ephemeral_turn(True)
    try:
        private = _traces_enabled_and_dir()
        assert "cache/private/" in str(private)
    finally:
        reset_ephemeral_turn(token)
    assert not private.exists()
