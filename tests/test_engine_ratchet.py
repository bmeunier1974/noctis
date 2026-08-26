"""The engine ratchet's *rule*: the tier split, and the version it is declared against.

The mechanics — building, loading and writing the record, the four-case check, what ``--write``
does with a verdict, the report — are shared by every ratchet and tested once in
``tests/test_ratchet.py``. What is left here is the policy this module exists to state: which tier
fails, which warns and passes, when ``ENGINE_VERSION`` has to move, and which drift ``--write``
refuses to record.

Every assertion is external behaviour — the status a comparison returns, the component and file
names it prints, the exit code the entrypoint gives CI. The comparison is pure over two records,
so a scenario is built by making a miniature repo in ``tmp_path``, recording it, editing exactly
one file, and recording it again — the same shape ``tests/test_engine_id.py`` uses.

The tier split is the contract: an edit to ``gates`` or ``backtest`` (the arbiter — what passes,
what a number means) **fails**; an edit to a searcher component **warns and passes**, because a
ratchet that blocks on a docstring edit gets disabled, and a disabled ratchet asserts nothing.

An arbiter move may be declared two ways, and the second one (#355) is the rest of this file: an
``ENGINE_VERSION`` bump says "these numbers are no longer comparable", and a dated entry at the top
of ``docs/engine-changelog.md`` naming the component under ``behaviour: unchanged`` says "this edit
was mechanical". Both still fail until the record catches up; what the declaration decides is
whether ``--write`` may be the thing that catches it up. How an entry is *read* is the shared
reader's, tested once in ``tests/test_changelog.py``; what is asserted here is what naming a
component there permits.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from noctis.observability import engine_change, engine_id, engine_ratchet, ratchet
from noctis.observability.changelog import newest_entry
from noctis.observability.engine_id import COMPONENT_PATHS, ENGINE_VERSION, fingerprint, tier_of
from noctis.observability.engine_ratchet import (
    CHANGELOG_PATH,
    RECORD_PATH,
    REGENERATE_COMMAND,
    build_record,
    check,
    compare_records,
    load_record,
    main,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET_SOURCE = REPO_ROOT / "src" / "noctis" / "observability" / "engine_ratchet.py"
SCRIPT = REPO_ROOT / "scripts" / "engine_fingerprint.py"

GATES_FILE = "src/noctis/champions/promotion.py"
GATES_SIBLING = "src/noctis/backtest/splits.py"
BACKTEST_FILE = "src/noctis/backtest/pipeline.py"
SCHEMA_FILE = COMPONENT_PATHS["schema"][0]

SEARCHER_EDITS = {
    "prompts": "src/noctis/research/prompt.py",
    "profiles": "mandate/profiles/aggressive.md",
    "seeds": "strategies/sma_crossover.py",
    "memory_seed": "MEMORY.seed.md",
    "research": "src/noctis/research/tools.py",
}


def _build_tree(root: Path) -> Path:
    """A miniature repo: every path the component map names, plus docs and a test file."""
    for paths in COMPONENT_PATHS.values():
        for rel in paths:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {rel}\nbody of {rel}\n")
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "architecture.md").write_text("# architecture\n")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_engine_ratchet.py").write_text("# a test\n")
    (root / "README.md").write_text("# Noctis\n")
    return root


def _record(root: Path, *, engine_version: int | None = None) -> dict:
    """The record for ``root``, optionally stamped with another declared version.

    A record carries the version that was declared when it was written, so overriding it is how
    a test expresses "this file was committed before the bump" without patching a constant.
    """
    record = build_record(root)
    if engine_version is not None:
        record["engine_version"] = engine_version
    return record


def _edit(root: Path, rel: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + "# a behavioural change\n")


def _tagged(result, tag: str) -> list[str]:
    """The names this verdict filed under one tier, in the order the report prints them."""
    return [drift.name for drift in result.drifts if drift.tag == tag]


# The page's own header, without entries — the shape ``docs/engine-changelog.md`` ships with.
CHANGELOG_HEADER = "# Engine changelog\n\nHow to read this file.\n\n"


def _write_changelog(root: Path, text: str) -> None:
    page = root / CHANGELOG_PATH
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(text, encoding="utf-8")


def _declare(root: Path, heading: str, note: str = "Mechanical only.") -> None:
    """Push one entry onto the front of the engine changelog, with the heading as given."""
    page = root / CHANGELOG_PATH
    existing = page.read_text(encoding="utf-8") if page.is_file() else CHANGELOG_HEADER
    older = existing.split("\n## ", 1)
    tail = "\n## " + older[1] if len(older) > 1 else ""
    _write_changelog(root, f"{CHANGELOG_HEADER}## {heading}\n\n{note}\n{tail}")


def _declare_no_op(root: Path, components: str) -> None:
    """The declaration itself: a dated entry naming the components under the no-op marker."""
    _declare(root, f"2026-02-02 — components: {components} — behaviour: unchanged")


def _stamp_committed_version(root: Path, version: int) -> None:
    """Rewrite the committed record's declared version — "this file predates the bump"."""
    path = root / RECORD_PATH
    record = json.loads(path.read_text())
    record["engine_version"] = version
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


# ── the policy module holds the rule and borrows the mechanics ────────────────────────────


def test_the_policy_module_runs_on_the_shared_mechanics(tmp_path):
    """The verdict types are the shared ones, and every verdict carries this policy's spec."""
    assert engine_ratchet.RatchetResult is ratchet.RatchetResult
    assert engine_ratchet.WriteOutcome is ratchet.WriteOutcome

    root = _build_tree(tmp_path)
    write_record(root)
    result = check(root)

    assert isinstance(result, ratchet.RatchetResult)
    assert result.spec is engine_ratchet.SPEC
    assert result.status == "ok"


def test_a_record_declares_the_engine_version_of_the_tree_it_states(tmp_path):
    """The declared comparison key is this policy's own field in the shared record."""
    record = build_record(_build_tree(tmp_path))

    assert record["engine_version"] == ENGINE_VERSION
    assert record["components"]["gates"]["digest"] == engine_id.fingerprint(tmp_path).digest(
        "gates"
    )


# ── the arbiter tier: fail ────────────────────────────────────────────────────────────────


def test_a_gates_edit_without_a_version_bump_fails_naming_the_component_and_the_file(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert not result.ok
    assert _tagged(result, "arbiter") == ["gates"]
    assert result.drifts[0].files == (GATES_FILE,)
    report = result.report()
    assert "gates" in report and GATES_FILE in report
    assert GATES_SIBLING not in report  # only what moved, never its untouched siblings
    assert "ENGINE_VERSION" in report


def test_a_backtest_edit_without_a_version_bump_fails_naming_the_component_and_the_file(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, BACKTEST_FILE)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert _tagged(result, "arbiter") == ["backtest"]
    assert result.drifts[0].files == (BACKTEST_FILE,)
    assert BACKTEST_FILE in result.report()


def test_bumping_the_version_without_regenerating_the_record_still_fails(tmp_path):
    """A bump is half the deal: the reviewer must also see which surface moved, in the diff."""
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root, engine_version=ENGINE_VERSION), committed)

    assert result.status == "fail"
    assert REGENERATE_COMMAND in result.report()


def test_an_arbiter_edit_with_a_bump_and_a_regenerated_record_passes(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)
    assert compare_records(_record(root, engine_version=ENGINE_VERSION - 1), committed).status == (
        "fail"
    )

    # The PR bumps ENGINE_VERSION and regenerates the committed file in the same commit.
    regenerated = _record(root, engine_version=ENGINE_VERSION)

    assert compare_records(regenerated, regenerated).status == "ok"


# ── the searcher tier: warn and pass ──────────────────────────────────────────────────────


@pytest.mark.parametrize(("component", "rel"), sorted(SEARCHER_EDITS.items()))
def test_a_searcher_edit_without_a_bump_warns_and_passes(tmp_path, component, rel):
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, rel)

    result = compare_records(_record(root), committed)

    assert result.status == "warn"
    assert result.ok  # visible, not blocking
    assert _tagged(result, "searcher") == [component]
    assert result.drifts[0].files == (rel,)
    report = result.report()
    assert component in report and rel in report
    assert not _tagged(result, "arbiter")


def test_a_searcher_edit_beside_an_arbiter_edit_still_fails(tmp_path):
    """The strict tier wins the verdict; the advisory tier is still named, and named second."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)
    _edit(root, SEARCHER_EDITS["prompts"])

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert _tagged(result, "searcher") == ["prompts"]
    assert [drift.name for drift in result.drifts] == ["gates", "prompts"]
    assert SEARCHER_EDITS["prompts"] in result.report()


def test_a_null_component_gaining_a_digest_is_searcher_tier_drift(tmp_path):
    """When the run-record schema module lands (#143), ``schema`` goes null → digest: a warn."""
    root = _build_tree(tmp_path)
    (root / SCHEMA_FILE).unlink()
    committed = _record(root)
    schema = root / SCHEMA_FILE
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text("SCHEMA_VERSION = 1\n")

    result = compare_records(_record(root), committed)

    assert result.status == "warn"
    assert _tagged(result, "searcher") == ["schema"]
    assert SCHEMA_FILE in result.report()


# ── staleness: the contributor is told to regenerate ──────────────────────────────────────


def test_a_record_that_no_longer_declares_this_version_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)

    result = compare_records(_record(root), committed)

    assert result.status == "fail"
    assert "stale" in result.report().lower()
    assert REGENERATE_COMMAND in result.report()


# ── what --write refuses: an arbiter move must arrive declared ────────────────────────────


def test_arbiter_drift_with_the_versions_in_agreement_is_what_write_refuses(tmp_path):
    """The loophole (#145): ``--write`` is the fix the failure itself recommends, and it rewrites
    every component at once — so if it absorbed an arbiter digest too, the ratchet's promise would
    reduce to a contributor noticing the failure before typing the command it printed."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root), committed)

    assert result.refuse_write


def test_searcher_drift_is_never_what_write_refuses(tmp_path):
    """The common case must stay one command, or the ratchet becomes a chore and gets disabled."""
    root = _build_tree(tmp_path)
    committed = _record(root)
    _edit(root, SEARCHER_EDITS["prompts"])

    result = compare_records(_record(root), committed)

    assert result.status == "warn"
    assert not result.refuse_write


def test_an_arbiter_move_whose_bump_is_declared_is_not_refused(tmp_path):
    """The bump is there; the record simply had not been regenerated with it yet."""
    root = _build_tree(tmp_path)
    committed = _record(root, engine_version=ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)

    result = compare_records(_record(root, engine_version=ENGINE_VERSION), committed)

    assert result.status == "fail"
    assert not result.refuse_write


# ── the declaration: a no-op entry names the component on the engine changelog ────────────


def test_a_record_declares_the_changelog_entry_it_was_written_against(tmp_path):
    """The engine record carries the prompt record's block, key for key: "arrived *after* the
    record" is checkable only because the record states the entry it was regenerated against."""
    root = _build_tree(tmp_path)
    _declare_no_op(root, "gates")

    block = build_record(root)["changelog"]

    assert set(block) == {"path", "heading", "digest", "declares"}
    assert block["path"] == CHANGELOG_PATH
    assert block["heading"] == "2026-02-02 — components: gates — behaviour: unchanged"
    assert block["declares"] == ["gates"]


def test_a_tree_with_no_engine_changelog_records_a_head_of_nulls(tmp_path):
    """A missing page is a tree with nothing declared, never an error."""
    assert build_record(_build_tree(tmp_path))["changelog"] == {
        "path": CHANGELOG_PATH,
        "heading": None,
        "digest": None,
        "declares": [],
    }


def test_a_no_op_entry_naming_the_drifted_component_is_declared_but_still_fails(tmp_path):
    """The bump-declared precedent: the arbiter tier fails until the record catches up. What the
    declaration changes is only whether ``--write`` may be what catches it up."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, GATES_FILE)
    _declare_no_op(root, "gates")

    result = check(root)

    assert result.status == "fail"
    assert _tagged(result, "arbiter, declared no-op") == ["gates"]
    assert result.drifts[0].files == (GATES_FILE,)
    assert not result.refuse_write
    report = result.report()
    assert "declared no-op arbiter drift: gates" in report
    assert "the record was not regenerated with it" in report
    assert REGENERATE_COMMAND in report


def test_regenerating_a_declared_no_op_writes_the_record_the_check_then_accepts(tmp_path):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, GATES_FILE)
    _declare_no_op(root, "gates")
    assert main(["--check", "--root", str(root)]) == 1

    assert main(["--write", "--root", str(root)]) == 0

    assert main(["--check", "--root", str(root)]) == 0
    written = load_record(root)
    assert written["components"]["gates"]["digest"] == fingerprint(root).digest("gates")
    assert written["engine_version"] == ENGINE_VERSION  # the declaration is not a bump
    assert written["changelog"]["declares"] == ["gates"]


def test_the_undeclared_refusal_names_the_bump_the_no_op_entry_and_restoring(tmp_path, capsys):
    """Three outs, one copy-paste away: bump, declare a no-op with the heading spelled out, or
    restore the behaviour. And ``--write`` still refuses the move nobody declared."""
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, GATES_FILE)
    capsys.readouterr()

    code = main(["--write", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 1
    assert "arbiter drift with no ENGINE_VERSION bump: gates" in out
    assert "bump ENGINE_VERSION" in out
    assert CHANGELOG_PATH in out
    assert '"## ' in out
    assert "components: gates — behaviour: unchanged" in out
    assert "restore the behaviour" in out
    assert "refusing to regenerate" in out


# ── two halves, both needed: name the component, and be new since the record ──────────────


def test_an_entry_that_was_already_the_head_when_the_record_was_written_declares_nothing(tmp_path):
    """Yesterday's no-op cannot license today's behaviour change to the same component."""
    root = _build_tree(tmp_path)
    _declare_no_op(root, "gates")
    write_record(root)  # the record now names that entry as the head it was written against
    _edit(root, GATES_FILE)

    result = check(root)

    assert result.status == "fail"
    assert result.refuse_write
    assert _tagged(result, "arbiter") == ["gates"]


def test_an_entry_naming_the_other_arbiter_component_declares_nothing(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, GATES_FILE)
    _declare_no_op(root, "backtest")

    result = check(root)

    assert result.refuse_write
    assert _tagged(result, "arbiter") == ["gates"]


def test_an_entry_that_omits_the_no_op_marker_declares_nothing(tmp_path):
    """The page may narrate a bump; only ``behaviour: unchanged`` declares one to this check."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, GATES_FILE)
    _declare(root, "2026-02-02 — components: gates")

    result = check(root)

    assert result.refuse_write
    assert _tagged(result, "arbiter") == ["gates"]


def test_a_fenced_heading_declares_nothing_and_a_real_entry_beneath_it_does(tmp_path):
    """The page documents its own heading in a fence, so the reader that skips fences is the
    shared one — a template is not a declaration."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, GATES_FILE)
    _write_changelog(
        root,
        CHANGELOG_HEADER
        + "```text\n## <YYYY-MM-DD> — components: gates — behaviour: unchanged\n```\n",
    )
    assert check(root).refuse_write

    _declare_no_op(root, "gates")

    assert not check(root).refuse_write


def test_a_no_op_entry_naming_a_searcher_component_is_inert(tmp_path):
    """Searcher drift never blocked, so there is nothing there for a declaration to lift."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, SEARCHER_EDITS["prompts"])
    _declare_no_op(root, "prompts")

    result = check(root)

    assert result.status == "warn"
    assert result.ok
    assert _tagged(result, "searcher") == ["prompts"]
    assert "declared no-op" not in result.report()


def test_a_declared_no_op_beside_an_undeclared_move_is_still_refused(tmp_path):
    """The strictest drift wins the verdict, and the report ranks what it prints: the undeclared
    arbiter move first, then the declared one, then the advisory tier."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, GATES_FILE)
    _edit(root, BACKTEST_FILE)
    _edit(root, SEARCHER_EDITS["prompts"])
    _declare_no_op(root, "backtest")

    result = check(root)

    assert result.refuse_write
    assert [drift.name for drift in result.drifts] == ["gates", "backtest", "prompts"]
    assert _tagged(result, "arbiter, declared no-op") == ["backtest"]
    assert main(["--write", "--root", str(root)]) == 1


def test_a_declared_no_op_beside_a_version_bump_prints_the_older_version_line_and_writes(tmp_path):
    """Both may have happened in one PR, so a no-op entry beside a bump is not a contradiction —
    and with the versions in disagreement the declaration decides nothing anyway."""
    root = _build_tree(tmp_path)
    write_record(root)
    _stamp_committed_version(root, ENGINE_VERSION - 1)
    _edit(root, GATES_FILE)
    _declare_no_op(root, "gates")

    result = check(root)

    assert result.status == "fail"
    assert "arbiter drift recorded under an older ENGINE_VERSION: gates" in result.report()
    assert _tagged(result, "arbiter") == ["gates"]
    assert not result.refuse_write
    assert main(["--write", "--root", str(root)]) == 0


# ── the report names the entry the check actually read ────────────────────────────────────


def test_the_report_names_the_changelog_entry_it_read_on_arbiter_drift(tmp_path):
    """ "I wrote one and it still fails" needs the entry the check actually saw."""
    root = _build_tree(tmp_path)
    _declare_no_op(root, "backtest")
    write_record(root)
    _edit(root, GATES_FILE)

    report = check(root).report()

    assert f"newest {CHANGELOG_PATH} entry: 2026-02-02 — components: backtest" in report


def test_the_report_says_so_when_the_engine_changelog_has_no_entry_at_all(tmp_path):
    root = _build_tree(tmp_path)
    _write_changelog(root, CHANGELOG_HEADER)
    write_record(root)
    _edit(root, GATES_FILE)

    assert f"newest {CHANGELOG_PATH} entry: none" in check(root).report()


def test_a_missing_record_still_names_the_changelog_entry_it_read(tmp_path):
    """The baseline is missing, so no drift is named — but "which entry did it read" is exactly
    as much the question here, and the answer must not depend on drift to hang it on."""
    root = _build_tree(tmp_path)
    _declare_no_op(root, "backtest")

    report = check(root).report()

    assert load_record(root) is None
    assert f"newest {CHANGELOG_PATH} entry: 2026-02-02 — components: backtest" in report


def test_searcher_only_drift_never_names_a_changelog_entry(tmp_path):
    """There is nothing to declare, and a check that always says something is one people skip."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, SEARCHER_EDITS["prompts"])

    result = check(root)

    assert result.status == "warn"
    assert result.footer == ()
    assert CHANGELOG_PATH not in result.report()


def test_the_policy_re_exports_none_of_the_shared_reader():
    """One parser for both ratchets, reached at one name. A convenience alias here would be a
    second name for the same reading, and the next contributor would have to work out which of
    the two this policy binds."""
    moved = ("ChangelogEntry", "newest_entry", "read_entry", "header", "declared_since", "footer")

    assert [name for name in moved if hasattr(engine_ratchet, name)] == []


# ── the exit code each tier gives CI ──────────────────────────────────────────────────────


def test_the_check_exits_non_zero_on_arbiter_drift_naming_the_component_and_file(tmp_path, capsys):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, GATES_FILE)

    code = main(["--check", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 1
    assert "gates" in out and GATES_FILE in out


def test_the_check_exits_zero_on_a_searcher_warning_naming_the_component_and_file(tmp_path, capsys):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, SEARCHER_EDITS["prompts"])

    code = main(["--check", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 0
    assert "prompts" in out and SEARCHER_EDITS["prompts"] in out
    assert "warn" in out.lower()


# ── one arbiter line, drawn once ──────────────────────────────────────────────────────────


def test_the_ratchet_and_the_resume_policy_read_one_arbiter_components_constant(tmp_path):
    """Two copies of that set would eventually disagree, and the disagreement would be silent.

    The CI ratchet (a change may not land) and the resume policy (a run may not continue) are the
    two enforcers of the one line, so this binds them **both ways**: they classify through the same
    function object, that function reads the one constant the resume policy also quotes by name,
    and — the part a refactor cannot fake — they return the same tier for every component when the
    whole engine has moved underneath them.
    """
    assert engine_ratchet.tier_of is engine_id.tier_of
    assert engine_change.tier_of is engine_id.tier_of
    assert engine_change.ARBITER_COMPONENTS is engine_id.ARBITER_COMPONENTS

    root = _build_tree(tmp_path)
    frozen = {
        "engine": {"engine_version": ENGINE_VERSION, "fingerprint": fingerprint(root).digests()}
    }
    committed = _record(root)
    for paths in COMPONENT_PATHS.values():
        _edit(root, paths[0])

    result = compare_records(_record(root), committed)
    ratchet_tiers = {drift.name: drift.tag for drift in result.drifts}
    policy = {
        change.component: change.tier
        for change in engine_change.engine_change(frozen, fingerprint(root)).components
    }
    assert policy == ratchet_tiers
    assert set(policy) == set(COMPONENT_PATHS)

    # No second literal anywhere a consumer (this ratchet, the resume policy) could read.
    sources = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "scripts").rglob("*"))
    literals = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in sources
        if path.suffix == ".py"
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if '"gates"' in line and '"backtest"' in line
    ]
    assert literals == ["src/noctis/observability/engine_id.py:54"], literals


# ── wiring and documentation ──────────────────────────────────────────────────────────────


def _development_section(title: str) -> str:
    """One ``## <title>`` section of the page that explains this ratchet to a contributor."""
    text = (REPO_ROOT / "docs" / "development.md").read_text(encoding="utf-8")
    start = text.find(f"## {title}\n")
    assert start >= 0, f"docs/development.md: the '{title}' section is gone — retarget this test"
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def _documented() -> str:
    """That section, whitespace-collapsed, so a sentence the page wraps still reads as one."""
    return " ".join(_development_section("The engine fingerprint ratchet").split())


def _assert_transcribed(printed: str) -> None:
    """Every line the tool printed appears in the page's transcript, bar the digest pair.

    A transcript is only worth printing if a contributor can compare it with their terminal, so
    it is quoted from the terminal. The one line that cannot be is the ``a -> b`` pair: those
    digests are the miniature tree's, where the page's are an illustration.
    """
    documented = _documented()
    for line in printed.splitlines():
        if "->" in line:
            continue
        assert " ".join(line.split()) in documented, line


def test_the_ratchet_runs_in_ci_and_in_pre_commit():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    hooks = (REPO_ROOT / ".pre-commit-config.yaml").read_text()

    assert "scripts/engine_fingerprint.py" in workflow
    assert "--check" in workflow
    assert "scripts/engine_fingerprint.py" in hooks


def test_the_pre_commit_hook_re_runs_the_check_when_the_engine_changelog_moves():
    """The declaration is half the input, so editing it has to re-run the check — otherwise a
    no-op entry lands in a commit the hook never looked at, exactly as the prompt hook watches
    its own page. A docs edit that declares nothing still never triggers it."""
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    pattern = hooks["engine-fingerprint"]["files"]

    assert re.search(pattern, CHANGELOG_PATH)
    assert re.search(pattern, GATES_FILE)
    assert re.search(pattern, RECORD_PATH)
    assert not re.search(pattern, "docs/development.md")


def test_the_operating_contract_names_both_ways_to_declare_an_arbiter_move():
    """AGENTS.md is what an agent reads before it edits anything, and its one-line summary of
    this command has to name both declarations — "declare it" is not a fix you can type."""
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    after = agents.split("python scripts/engine_fingerprint.py", 1)[1]
    comment = after.split("python scripts/prompt_fingerprint.py", 1)[0]

    assert "ENGINE_VERSION" in comment
    assert CHANGELOG_PATH in comment


def test_the_regeneration_command_is_documented():
    development = (REPO_ROOT / "docs" / "development.md").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()

    assert REGENERATE_COMMAND in development
    assert "scripts/engine_fingerprint.py" in agents
    assert RECORD_PATH in development


def test_the_write_refusal_is_documented_where_a_contributor_reads_the_rule():
    """A guard nobody has been told about reads as a broken tool, so the one place the ratchet is
    explained to a contributor has to say that ``--write`` will not record an undeclared move."""
    development = (REPO_ROOT / "docs" / "development.md").read_text()
    script = (REPO_ROOT / "scripts" / "engine_fingerprint.py").read_text()

    assert "refuses to regenerate" in development
    assert "declared" in development
    assert "refuses" in script


def test_the_declared_no_op_path_is_documented_where_the_bump_path_is():
    """A contributor who reads only this page must be able to reach *both* declarations from it:
    a rule stated in half is the one people work around."""
    section = _development_section("The engine fingerprint ratchet")

    assert CHANGELOG_PATH in section
    assert "no-op" in section
    assert "declared" in section
    assert "behaviour: unchanged" in section


def test_the_documented_refusal_transcript_names_the_three_outs_in_the_tools_own_words(
    tmp_path, capsys
):
    """The page's ``--write`` transcript is the terminal's: the refusal names three outs — bump,
    declare a no-op, restore — and names the changelog entry the check actually read."""
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, GATES_FILE)
    capsys.readouterr()

    assert main(["--write", "--root", str(root)]) == 1

    _assert_transcribed(capsys.readouterr().out)


def test_the_documented_declared_no_op_transcript_is_the_tools_own_words(tmp_path, capsys):
    """And the page shows the other half of the same story: a declared move still fails until the
    record catches up, filed under its own tag, with regenerating as the advice."""
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, BACKTEST_FILE)
    _declare(root, "2026-08-25 — components: backtest — behaviour: unchanged")
    capsys.readouterr()

    assert main(["--check", "--root", str(root)]) == 1

    _assert_transcribed(capsys.readouterr().out)


def test_the_page_says_what_declaring_a_no_op_does_not_lift():
    """The declaration lifts the ``--write`` refusal and nothing else: the digest still moved, so
    the resume policy and ``comparable_key`` keep partitioning on it. Over-partitioning is the
    accepted failure, and a page that left that out would read as "a no-op costs nothing"."""
    section = _development_section("The engine fingerprint ratchet")

    assert "resume policy" in section
    assert "comparable_key" in section
    assert "over-partition" in section.lower()


def test_the_tier_rule_is_documented_in_the_module_docstring():
    """The rule (fail on arbiter, warn on searcher, stale is what you are told to fix, and
    ``--write`` will not launder an undeclared arbiter move) is the contract; it is written where
    the next reader of the code will be."""
    docstring = RATCHET_SOURCE.read_text().split('"""')[1].lower()

    assert "arbiter" in docstring and "searcher" in docstring
    assert "regenerate" in docstring
    assert "--write" in docstring and "refus" in docstring


def test_the_script_is_a_shim_a_guard_and_one_main_import():
    """``scripts/engine_fingerprint.py`` runs *before* the package is importable, which is the
    only reason it exists: it holds the ``sys.path`` shim, the ``ImportError`` guard and the one
    import of ``main`` — no policy, no argument parsing, nothing a module could hold instead."""
    module = ast.parse(SCRIPT.read_text())
    docstring, *statements = module.body

    assert isinstance(docstring, ast.Expr) and isinstance(docstring.value, ast.Constant)
    assert [type(node).__name__ for node in statements] == [
        "Import",  # sys, for the shim
        "ImportFrom",  # pathlib.Path, for the shim
        "Assign",  # _SRC
        "If",  # the sys.path shim
        "Try",  # the ImportError guard
        "If",  # the __main__ dispatch
    ]

    guard = statements[4]
    assert isinstance(guard, ast.Try)
    assert [ast.unparse(node) for node in guard.body] == [
        "from noctis.observability.engine_ratchet import main"
    ]

    dispatch = statements[5]
    assert isinstance(dispatch, ast.If)
    assert ast.unparse(dispatch.test) == "__name__ == '__main__'"


def test_the_script_docstring_names_what_write_refuses():
    """A guard nobody has been told about reads as a broken tool, and the script is where an
    operator lands first — so its docstring keeps the refusal sentence."""
    docstring = ast.get_docstring(ast.parse(SCRIPT.read_text())) or ""

    assert "refuses" in docstring
    assert "--write" in docstring
    assert "arbiter" in docstring and "ENGINE_VERSION" in docstring


# ── the page a declaration is written on ──────────────────────────────────────────────────


CHANGELOG_PAGE = REPO_ROOT / CHANGELOG_PATH

# The clause of the shared changelog grammar this policy reads names off.
COMPONENTS_CLAUSE = "components"


def test_the_newest_changelog_entry_names_only_arbiter_components():
    """A declaration a machine cannot resolve to a component declares nothing — a typo'd or
    searcher name would leave real drift undeclared under a heading that reads as if it covered
    it. Holds with no entry on the page at all, which is the state it ships in."""
    entry = newest_entry(CHANGELOG_PAGE.read_text(encoding="utf-8"))
    names = () if entry is None else entry.names(COMPONENTS_CLAUSE)

    assert set(names) <= engine_id.ARBITER_COMPONENTS


def test_the_engine_changelog_page_carries_its_grammar_and_no_entries_yet():
    """The page ships with its header alone: the grammar lives in a fence the reader skips, so
    the template that documents a declaration is not itself one."""
    text = CHANGELOG_PAGE.read_text(encoding="utf-8")

    assert newest_entry(text) is None
    assert "components: <component>[, <component>…] — behaviour: unchanged" in text


@pytest.mark.parametrize("component", sorted(COMPONENT_PATHS))
def test_the_page_names_every_component_with_its_tier_and_the_files_its_digest_covers(component):
    """A declaration a machine reads has to be one a human can write: the page spells each
    component as the map does, says which tier it is on, and names what its digest covers."""
    row = next(
        line
        for line in CHANGELOG_PAGE.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"| `{component}`")
    )

    assert tier_of(component) in row
    for rel in COMPONENT_PATHS[component]:
        assert f"`{rel}`" in row, (component, rel)


def test_the_page_states_what_qualifies_as_a_no_op_what_never_does_and_the_reviewers_bar():
    """The claim is a human's word, so the bar it is judged against is on the page beside it."""
    text = CHANGELOG_PAGE.read_text(encoding="utf-8").lower()

    for mechanical in ("rename", "import path", "docstring", "type annotation", "pass-through"):
        assert mechanical in text
    for behavioural in ("branch", "constant", "default", "formula", "threshold"):
        assert behavioural in text
    assert "golden" in text and "fixture" in text


def test_the_committed_records_changelog_head_is_what_this_tree_reads():
    """The drift check compares *components* only, so the record's changelog block — the second
    half of the rule — is unpinned by it: a parser that read this page differently would leave
    the committed head stale and nothing would say so. This is that pin, field by field."""
    committed = json.loads((REPO_ROOT / RECORD_PATH).read_text())["changelog"]
    read_now = build_record(REPO_ROOT)["changelog"]

    assert committed["path"] == read_now["path"] == CHANGELOG_PATH
    assert committed["heading"] == read_now["heading"]
    assert committed["digest"] == read_now["digest"]
    assert committed["declares"] == read_now["declares"]


def test_the_declared_no_op_case_is_documented_in_the_module_docstring():
    """The second way to declare an arbiter move is written where the next reader of the code
    will be, beside the bump it is an alternative to."""
    docstring = RATCHET_SOURCE.read_text().split('"""')[1].lower()

    assert "no-op" in docstring
    assert CHANGELOG_PATH in docstring
    assert "declar" in docstring
