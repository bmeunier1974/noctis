"""PR2 — LLM ideation seam: mocked-client minting, validation/parity admission, loop
folding, and the no-key graceful fallback (research runs seed-families-only). Ideation
talks to the provider-neutral :class:`~noctis.research.llm.LLMClient`, so the fakes here
implement ``complete`` and return :class:`~noctis.research.llm.Turn`."""

from __future__ import annotations

import itertools
import json

import pytest

from noctis.backtest.scorecard import Scorecard
from noctis.champions import ChampionRegistry, PromotionRules
from noctis.config.settings import IdeationConfig, Settings
from noctis.engine import run_research
from noctis.memory import InMemoryMemory
from noctis.research import Capabilities, IdeationContext, Ideator, build_ideator, propose_specs
from noctis.research.llm import ToolCall, Turn
from noctis.strategies import Candidate, CandidateProposer
from noctis.strategies.families import SEED_FAMILIES, FamilyRegistry

RULES = PromotionRules(champion_count=3, max_gap=1.0, min_test_metric=0.0)


# ── A canned, valid, parity-passing spec (SMA crossover) and a broken one ────────────────
def _valid_spec(spec_id: str) -> dict:
    return {
        "version": 1,
        "id": spec_id,
        "name": "minted SMA crossover",
        "sources": [{"id": "src", "schema": "ohlcv-1m"}],
        "parameters": [
            {"id": "fast", "kind": "int", "value": 8},
            {"id": "slow", "kind": "int", "value": 21},
        ],
        "features": [
            {"id": "f", "kind": "sma", "input": "src", "period": "fast"},
            {"id": "s", "kind": "sma", "input": "src", "period": "slow"},
        ],
        "signals": [
            {"id": "en", "kind": "condition", "op": ">", "a": "f", "b": "s"},
            {"id": "ex", "kind": "condition", "op": "<=", "a": "f", "b": "s"},
        ],
        "entries": [{"id": "e", "enter": "en", "exit": "ex"}],
        "optimizations": [
            {
                "id": "opt",
                "parameters": [
                    {"param": "fast", "type": "int", "min": 3, "max": 20, "step": 1},
                    {"param": "slow", "type": "int", "min": 15, "max": 60, "step": 1},
                ],
            },
        ],
    }


def _dangling_spec(spec_id: str) -> dict:
    """Schema-valid shape but a signal references a non-existent feature → model_validate raises."""
    spec = _valid_spec(spec_id)
    spec["signals"][0]["b"] = "does_not_exist"
    return spec


# ── A fake LLMClient returning the emit tool call as a seam Turn ─────────────────────────
def _emit_turn(strategies: list[dict]) -> Turn:
    call = ToolCall(id="tc_1", name="emit_strategies", arguments={"strategies": strategies})
    return Turn(text="", tool_calls=[call], stop_reason="tool_use", usage={})


class FakeClient:
    def __init__(self, strategies: list[dict], *, capabilities: Capabilities | None = None):
        self._strategies = strategies
        self.capabilities = capabilities or Capabilities()
        self.calls = 0
        self.last_kwargs: dict = {}

    def complete(self, *, system, tools, messages, max_tokens, tool_choice=None, on_delta=None):
        self.calls += 1
        self.last_kwargs = {
            "system": system,
            "tools": tools,
            "messages": messages,
            "max_tokens": max_tokens,
            "tool_choice": tool_choice,
        }
        return _emit_turn(self._strategies)


class PausingClient(FakeClient):
    """First completion pauses mid-turn (server-tool loop hit its cap); the resume emits."""

    def complete(self, **kwargs):
        turn = super().complete(**kwargs)
        if self.calls == 1:
            return Turn(
                text="",
                tool_calls=[],
                stop_reason="pause_turn",
                usage={},
                assistant_message={"role": "assistant", "content": "searching…"},
            )
        return turn


class SearchingClient(FakeClient):
    """First completion calls the client-side web_search tool; the resume emits."""

    def complete(self, **kwargs):
        turn = super().complete(**kwargs)  # records kwargs; returns the emit turn
        if self.calls == 1:
            args = {"query": "momentum factor", "max_results": 3}
            return Turn(
                text="",
                tool_calls=[ToolCall(id="ws_1", name="web_search", arguments=args)],
                stop_reason="tool_use",
                usage={},
                assistant_message={
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "ws_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": json.dumps(args)},
                        }
                    ],
                },
            )
        return turn


@pytest.fixture
def families():
    """A per-test registry — minted families land here and die with the test."""
    return FamilyRegistry()


class _Reg:
    def __init__(self, champions=None):
        self._c = champions or []

    def list(self):
        return self._c


# ── 1. propose_specs: valid admitted, invalid dropped, forced tool-use call shape ────────
def test_propose_specs_admits_valid_drops_invalid():
    client = FakeClient([_valid_spec("minted_ok"), _dangling_spec("minted_bad")])
    specs = propose_specs(context=IdeationContext(), n=2, client=client)
    assert [s.id for s in specs] == ["minted_ok"]
    # The emit tool is forced (OpenAI format; LiteLLM translates per provider).
    kw = client.last_kwargs
    assert kw["tool_choice"] == {"type": "function", "function": {"name": "emit_strategies"}}


def test_propose_specs_web_search_offers_tool_unforced():
    """With web_search on (and the capability present), the emit tool is offered — not forced —
    alongside the server web tool."""
    client = FakeClient(
        [_valid_spec("minted_web")], capabilities=Capabilities(server_web_search=True)
    )
    specs = propose_specs(context=IdeationContext(), n=1, client=client, web_search=True)
    assert [s.id for s in specs] == ["minted_web"]
    kw = client.last_kwargs
    # emit tool can no longer be forced (that would skip the search) — offered instead.
    assert kw["tool_choice"] is None
    tool_types = {t.get("type", t.get("name")) for t in kw["tools"]}
    assert "web_search_20260209" in tool_types
    assert "emit_strategies" in tool_types  # emit tool carries a name, not a type


def test_propose_specs_web_search_falls_back_to_client_sidecar():
    """A provider with no server-side web search (OpenAI, any local backend) no longer degrades
    to grounding-less ideation: the client-side sidecar tool is offered (emit no longer forced)
    and the search prompt guidance stays; the model emits at once, so nothing is dispatched."""
    client = FakeClient([_valid_spec("minted_grounded")])  # default caps: no server search
    specs = propose_specs(context=IdeationContext(), n=1, client=client, web_search=True)
    assert [s.id for s in specs] == ["minted_grounded"]
    kw = client.last_kwargs
    assert kw["tool_choice"] is None  # offered, not forced (forcing emit would skip the search)
    assert {t.get("name") for t in kw["tools"]} == {"emit_strategies", "web_search"}
    ws = next(t for t in kw["tools"] if t.get("name") == "web_search")
    assert "input_schema" in ws and "type" not in ws  # client function tool, not a server tool
    assert "web_search" in kw["messages"][0]["content"]  # search guidance retained


def test_propose_specs_client_web_search_round_trip(monkeypatch):
    """On a keyless backend, a client-side web_search tool call is dispatched to the sidecar,
    its result appended as a tool message, and the resumed turn emits — grounding without a
    server capability, and without touching the real network in the test."""
    from noctis.research import websearch

    captured: dict = {}

    def fake_search(query, max_results=5):
        captured.update(query=query, n=max_results)
        return {"query": query, "results": [{"title": "M", "url": "http://x", "snippet": "s"}]}

    monkeypatch.setattr(websearch, "search", fake_search)

    client = SearchingClient([_valid_spec("minted_after_search")])  # no server search
    specs = propose_specs(context=IdeationContext(), n=1, client=client, web_search=True)

    assert [s.id for s in specs] == ["minted_after_search"]
    assert client.calls == 2  # the search turn, then the emit turn
    assert captured == {"query": "momentum factor", "n": 3}  # the sidecar was actually called
    # The resumed request replays the assistant search turn + a tool-role result with the id.
    resumed = client.last_kwargs["messages"]
    assert resumed[1]["role"] == "assistant"
    assert resumed[2]["role"] == "tool" and resumed[2]["tool_call_id"] == "ws_1"


def test_propose_specs_resumes_a_paused_server_tool_turn():
    """``pause_turn`` (a server-tool loop hitting its cap) is resumed by re-sending the
    assistant turn verbatim — the same idiom as the agent research loop."""
    client = PausingClient(
        [_valid_spec("minted_resumed")], capabilities=Capabilities(server_web_search=True)
    )
    specs = propose_specs(context=IdeationContext(), n=1, client=client, web_search=True)
    assert [s.id for s in specs] == ["minted_resumed"]
    assert client.calls == 2
    # The resumed request replays the paused assistant turn after the original prompt.
    assert client.last_kwargs["messages"][1] == {"role": "assistant", "content": "searching…"}


# ── 2. Ideator.run mints: registers family, grows proposer, records a finding, persists ──
def test_ideator_mints_registers_and_feeds_back(tmp_path, families):
    client = FakeClient([_valid_spec("minted_sma")])
    proposer = CandidateProposer(families, seed=0)
    memory = InMemoryMemory()
    ideator = Ideator(
        client=client,
        config=IdeationConfig(cadence=1, specs_per_round=1),
        registry=_Reg(),
        families=families,
        proposer=proposer,
        memory=memory,
        state_dir=tmp_path,
    )

    minted = ideator.run(0)

    assert minted == ["minted_sma"]
    assert "minted_sma" in proposer.rotation  # enters the proposal rotation
    assert "minted_sma" in families  # registered as a family
    assert any("minted_sma" in f for f in memory.findings())  # memory feedback
    assert (tmp_path / "specs.json").is_file()  # persisted for restart survival
    # The minted family is now proposable + buildable through the ordinary pipeline.
    cand = Candidate("minted_sma", {"fast": 8, "slow": 21})
    assert cand.build(families) is not None


# ── 3. Cadence gating: mints on the seed round + every `cadence`, idle otherwise ─────────
def test_ideator_respects_cadence(tmp_path, families):
    counter = itertools.count()
    client = FakeClient([])  # no specs; we only assert *when* it calls the API

    def specs_for_round():
        # unique-id spec per successful round so registration never collides
        return [_valid_spec(f"minted_{next(counter)}")]

    ideator = Ideator(
        client=client,
        config=IdeationConfig(cadence=3, specs_per_round=1),
        registry=_Reg(),
        families=families,
        proposer=CandidateProposer(families, seed=0),
        memory=InMemoryMemory(),
        state_dir=tmp_path,
    )
    for i in range(7):
        ideator.run(i)
    # iterations 0, 3, 6 hit the API; 1,2,4,5 are gated out.
    assert client.calls == 3


# ── 4. Loop folds minted names into the summary (via run_research's ideate seam) ─────────
def test_run_research_folds_minted_specs(tmp_path):
    def evaluate_fn(cand: Candidate) -> Scorecard:
        return Scorecard(family=cand.family, params=cand.params, stage="prefilter_rejected")

    def ideate(i: int) -> list[str]:
        return ["spec_seed"] if i == 0 else []

    summary = run_research(
        proposer=CandidateProposer(seed=1),
        evaluate_fn=evaluate_fn,
        registry=ChampionRegistry(tmp_path / "c.json", capacity=3),
        rules=RULES,
        memory=InMemoryMemory(),
        budget_minutes=60.0,
        max_iterations=3,
        ideate=ideate,
    )
    assert summary.minted_specs == ["spec_seed"]
    assert summary.iterations == 3


# ── 5. Collision guard: an id matching an existing family is skipped, never clobbered ────
def test_ideator_skips_colliding_family_id(tmp_path, families):
    from noctis.strategies import SmaCrossover

    client = FakeClient([_valid_spec("sma_crossover"), _valid_spec("minted_fresh")])
    proposer = CandidateProposer(families, seed=0)
    ideator = Ideator(
        client=client,
        config=IdeationConfig(cadence=1, specs_per_round=2),
        registry=_Reg(),
        families=families,
        proposer=proposer,
        memory=InMemoryMemory(),
        state_dir=tmp_path,
    )

    minted = ideator.run(0)

    assert minted == ["minted_fresh"]  # the collision was dropped
    assert families.get_class("sma_crossover") is SmaCrossover  # seed class not clobbered
    assert proposer.rotation.count("sma_crossover") == 1  # seed rotation slot, not doubled
    persisted = json.loads((tmp_path / "specs.json").read_text())["specs"]
    assert set(persisted) == {"minted_fresh"}  # collision never persisted
    # The prompt warns the model off the taken ids in the first place.
    prompt = client.last_kwargs["messages"][0]["content"]
    assert "sma_crossover" in prompt


def test_ideator_identical_respec_is_idempotent(tmp_path, families):
    proposer = CandidateProposer(families, seed=0)
    memory = InMemoryMemory()
    client = FakeClient([_valid_spec("minted_once")])
    ideator = Ideator(
        client=client,
        config=IdeationConfig(cadence=1, specs_per_round=1),
        registry=_Reg(),
        families=families,
        proposer=proposer,
        memory=memory,
        state_dir=tmp_path,
    )

    assert ideator.run(0) == ["minted_once"]
    # Claude re-proposes the identical spec next round: no error, no duplicate mint.
    assert ideator.run(0) == []
    assert "minted_once" in families
    assert proposer.rotation.count("minted_once") == 1
    assert sum("minted_once" in f for f in memory.findings()) == 1


# ── 6. No-key fallback: build_ideator without a key mints nothing; research runs seed-only ─
def test_no_key_ideator_runs_seed_only(tmp_path):
    settings = Settings()
    settings.anthropic_api_key = None
    ideator = build_ideator(
        settings=settings,
        registry=_Reg(),
        families=FamilyRegistry(),
        proposer=CandidateProposer(seed=0),
        memory=InMemoryMemory(),
        state_dir=tmp_path,
    )
    assert ideator.client is None
    assert ideator.run(0) == [] and ideator.run(5) == []

    def evaluate_fn(cand: Candidate) -> Scorecard:
        return Scorecard(family=cand.family, params=cand.params, stage="prefilter_rejected")

    summary = run_research(
        proposer=CandidateProposer(seed=2),
        evaluate_fn=evaluate_fn,
        registry=ChampionRegistry(tmp_path / "c.json", capacity=3),
        rules=RULES,
        memory=InMemoryMemory(),
        budget_minutes=60.0,
        max_iterations=2,
        ideate=ideator.run,
    )
    assert summary.minted_specs == []  # seed-families-only, no error
    assert set(CandidateProposer(seed=2).rotation) == set(SEED_FAMILIES)


# ── 7. Same provider seam grammar as research.model (#10) ───────────────────────────────────
def test_ideation_model_rides_the_provider_seam(tmp_path, monkeypatch):
    """``ideation.model`` accepts the ``provider/model`` grammar and builds through the same
    :func:`client_for` seam as agent research: any provider with a key (or a keyless local
    backend) gets a client; a provider missing its key stays cleanly disabled."""
    import sys
    import types

    # A stand-in litellm module so the builder is deterministic without the [llm] extra.
    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))

    def _build(model, **keys):
        s = Settings(**keys)
        s.ideation.model = model
        return build_ideator(
            settings=s,
            registry=_Reg(),
            families=FamilyRegistry(),
            proposer=CandidateProposer(seed=0),
            memory=InMemoryMemory(),
            state_dir=tmp_path,
        )

    # Prefixed Anthropic model → client built from the anthropic key, model kept verbatim
    # (LiteLLM wants the provider/model form), Sonnet thinking pinned off as everywhere else.
    anth = _build("anthropic/claude-sonnet-5", anthropic_api_key="ak")
    assert anth.client is not None and anth.client.model == "anthropic/claude-sonnet-5"
    assert anth.client._thinking == {"type": "disabled"}
    # Bare legacy id keeps working unchanged.
    assert _build("claude-opus-4-8", anthropic_api_key="ak").client.model == "claude-opus-4-8"
    # Ideation is provider-neutral now: a non-Anthropic model with its key builds a client …
    assert _build("openai/gpt-5.4", openai_api_key="ok").client is not None
    # … and a provider whose key is missing stays cleanly disabled.
    assert _build("openai/gpt-5.4", anthropic_api_key="ak").client is None


def test_ideation_disabled_by_config_switch(tmp_path, monkeypatch):
    """``ideation.enabled: false`` is clientless even with a key + the extra present."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "litellm", types.ModuleType("litellm"))
    s = Settings(anthropic_api_key="ak")
    s.ideation.enabled = False
    ideator = build_ideator(
        settings=s,
        registry=_Reg(),
        families=FamilyRegistry(),
        proposer=CandidateProposer(seed=0),
        memory=InMemoryMemory(),
        state_dir=tmp_path,
    )
    assert ideator.client is None and ideator.run(0) == []
