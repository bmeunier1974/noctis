"""Golden hashes for what the model is actually told (epic #255, story #256).

Four rendered texts are pinned here by length + SHA-256: the FORMULATE, DECIDE and DISCOVER
briefings (:mod:`noctis.research.briefings`) over the briefing tests' populated fixture at a
huge context window, and the conversation system prompt (:mod:`noctis.research.prompt`) over
the tools tests' toolbox builder, at both ``prefix_trim`` values — five fingerprints in one
place.

They exist because the ``ResearchToolbox`` seam refactor moves *how* these renderers read
their facts. Moving a reader must not change one byte of what the model is told, and a diff
over a 20k-character prompt is unreadable, so the byte identity is asserted directly. These
goldens were taken on the branch's first commit, while the code still renders exactly what
``main`` renders; every later story in the epic keeps them green. Precedent and comment style:
``tests/test_strategy_author.py``'s ``_GOLDEN_SYSTEM_PROMPT_SHA256``.

What is NOT pinned here: the shipped ``TEMPLATE.py`` body the strategy-file contract embeds. The
tools tests' builder hands the toolbox an empty seeds tier, so the contract renders its
``(none)`` branch — deliberately, because the template's own bytes are already locked by
``tests/test_strategy_author.py`` (over a real install's seeds) and by the prompt-asset ratchet.
These goldens pin the *composition*: which facts appear, in what order, in what framing.

The fingerprints are machine-stable, not merely stable here: every input is seeded
deterministically (fixed bar tapes, an in-memory champion board and memory, a fixed session
ledger), and the library index carries names and header fields, never paths — so no tmp path,
wall clock, or dict ordering reaches the bytes. The path half of that is asserted structurally
below: the same render under two different tmp roots is byte-identical.

REGENERATING (a *declared* prompt change only — never to make a red test green): run

    uv run python -m tests.test_prompt_goldens
    uv run pytest -s tests/test_prompt_goldens.py     # the same thing, longhand

which prints a paste-ready ``_GOLDENS`` block from the same render the assertion reads, and
paste it over the one below. A prompt change belongs in the same commit as a new
``docs/prompt-changelog.md`` entry and a regenerated ``prompt_fingerprint.json``
(``uv run python scripts/prompt_fingerprint.py --write``).
"""

from __future__ import annotations

import hashlib

import pytest

from noctis.research.briefings import decide_briefing, formulate_briefing
from noctis.research.prompt import build_system_prompt
from tests.test_briefings import _HUGE, _discover, _populate
from tests.test_research_tools import _make_toolbox

# Fixed session arguments for the system prompt — the numbers the agent tests already use, so
# the golden reads like a real session's prefix rather than an invented one.
_BUDGET_MINUTES = 60.0
_MAX_ITERATIONS = 40
_DECIDE_SUBJECT = "probe"  # the strategy the populated fixture journals trials for

# The advisory memory the system-prompt toolbox carries. The tools tests' builder ships an empty
# memory, and `prefix_trim` trims *only* the advisory tail (findings + dead ends, 5 classes
# instead of 20) — so without a tail past the trimmed cap both prefix_trim goldens would be the
# same bytes and neither would pin anything. Seeded here, in the test, deliberately: the fixture
# owns its inputs, production code owns the rendering.
_SEEDED_FINDINGS = 8
_SEEDED_DEAD_ENDS = 6

# The pinned bytes: name → (length, sha256 of the utf-8 text). Regenerate with the command in
# the module docstring; see it for what regenerating obliges you to also do.
_GOLDENS: dict[str, tuple[int, str]] = {
    "formulate_briefing": (
        5403,
        "331353c0acbaaf6b8a45a98ebcfe24180722b45ee4fb7d6eaffa79ccb11d835d",
    ),
    "decide_briefing": (
        5224,
        "1bf144c50d5102c2f6b42d815f1661ae9f63637f23bfb67a7a42448c39ff4239",
    ),
    "discover_briefing": (
        2989,
        "309ba37a9d132bac2f06eb8106f3d54d9f9fe9d306da8c7fcb59b1469039ca8d",
    ),
    "system_prompt": (
        12716,
        "b90c86e2bae34e04961ba5c96e701174282c3f1f26b7a40327608f7a4dabdfd5",
    ),
    "system_prompt_prefix_trim": (
        12499,
        "baeab7cbd06be8e655e08b28b13f21d4f587a305b56c4c5451cbf8a1251c1c8c",
    ),
}


def _fingerprint(text: str) -> tuple[int, str]:
    return len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_toolbox(root):
    """The tools tests' toolbox builder, plus a deterministic advisory memory tail."""
    box = _make_toolbox(root)
    for i in range(_SEEDED_FINDINGS):
        box.memory.append_finding(f"finding-{i}: a lesson the session recorded")
    for i in range(_SEEDED_DEAD_ENDS):
        box.memory.record_rejected(f"deadend_{i}", {"lookback": i}, reason=f"no edge past cost {i}")
    return box


def _in_own_workspace(root, monkeypatch, build):
    """Build one fixture under a workspace of its own, and return it.

    A toolbox's WRITABLE library tiers (``__tmp``/``champions``) hang off ``run_dir`` — i.e. off
    ``NOCTIS_WORKSPACE``, which conftest pins once per test — not off the ``strategies_dir`` the
    builder is handed. Two fixtures built under one workspace would therefore SHARE a library
    tier, and each one's rendered index would depend on which was built first. A workspace per
    fixture is what makes a render a pure function of its own inputs.
    """
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NOCTIS_WORKSPACE", str(root / "workspace"))
    return build(root)


def _render(root, monkeypatch) -> dict[str, str]:
    """The five texts, rendered fresh under ``root``. Nothing but the directory the fixtures are
    built in varies between calls — and that must not reach the bytes."""
    box, ledger, mandate = _in_own_workspace(root / "briefings", monkeypatch, _populate)
    prompt_box = _in_own_workspace(root / "prompt", monkeypatch, _prompt_toolbox)
    return {
        "formulate_briefing": formulate_briefing(
            box, ledger, mandate=mandate, context_window=_HUGE
        ),
        "decide_briefing": decide_briefing(
            box, ledger, _DECIDE_SUBJECT, mandate=mandate, context_window=_HUGE
        ),
        "discover_briefing": _discover(box, ledger, mandate, context_window=_HUGE),
        "system_prompt": build_system_prompt(
            prompt_box,
            budget_minutes=_BUDGET_MINUTES,
            max_iterations=_MAX_ITERATIONS,
            mandate=None,
            prefix_trim=False,
        ),
        "system_prompt_prefix_trim": build_system_prompt(
            prompt_box,
            budget_minutes=_BUDGET_MINUTES,
            max_iterations=_MAX_ITERATIONS,
            mandate=None,
            prefix_trim=True,
        ),
    }


def _paste_ready(fingerprints: dict[str, tuple[int, str]]) -> str:
    """The current fingerprints as the literal block above — already in the shape ``ruff format``
    leaves alone, so regenerating is a paste and nothing else."""
    body = "\n".join(
        f'    "{k}": (\n        {n},\n        "{sha}",\n    ),'
        for k, (n, sha) in fingerprints.items()
    )
    return f"_GOLDENS: dict[str, tuple[int, str]] = {{\n{body}\n}}"


@pytest.fixture(autouse=True)
def _in_process_gate(fast_gate):
    """The fixtures write strategy files through the real write gate; this module pins rendered
    text, not subprocess isolation, so the gate runs through the library's in-process seam."""


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    return _render(tmp_path, monkeypatch)


def test_the_briefings_and_system_prompt_render_their_golden_bytes(rendered):
    # Every fingerprint at once, so a refactor that shifts two texts fails naming both.
    actual = {name: _fingerprint(text) for name, text in rendered.items()}
    # The regeneration helper: printed live under `-s`, and replayed by pytest on failure.
    print(f"\n{_paste_ready(actual)}")

    assert actual == _GOLDENS


def test_the_same_render_under_a_different_tmp_root_is_byte_identical(tmp_path, monkeypatch):
    # The machine-stability half a hash cannot tell you by itself: nothing about WHERE the
    # fixtures live reaches the bytes. A leaked tmp path would show up here, not two months
    # from now on someone else's checkout.
    assert _render(tmp_path / "one", monkeypatch) == _render(tmp_path / "two", monkeypatch)


def test_both_prefix_trim_goldens_pin_a_different_prompt(rendered):
    # Guards the goldens against going degenerate: `prefix_trim` must still bite on this
    # fixture, or the second system-prompt golden pins the first one twice.
    assert rendered["system_prompt"] != rendered["system_prompt_prefix_trim"]
    assert len(rendered["system_prompt_prefix_trim"]) < len(rendered["system_prompt"])


if __name__ == "__main__":  # `python -m tests.test_prompt_goldens` → print the current values
    raise SystemExit(pytest.main([__file__, "-s", "-q", "-p", "no:cacheprovider"]))
