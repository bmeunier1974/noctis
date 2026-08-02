"""The prompt ratchet (#183): a prompt change cannot land undeclared.

Every assertion here is external behaviour — the status a comparison returns, the site and file
names it prints, the exit code the entrypoint gives CI, what the committed record holds, whether a
file was written. The comparison is pure over two records, so a scenario is a miniature repo in
``tmp_path``: record it, edit one prompt asset, optionally declare the change in the changelog,
and record it again — the shape ``tests/test_engine_ratchet.py`` uses.

The declared-change rule is the contract: drift fails until the newest changelog entry *names the
drifted site* **and** that entry arrived after the committed record was written. Both halves get
tests, because either one alone is a rubber stamp — a nameless entry declares nothing, and an
entry that was already the head when the record was written declares nothing new.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noctis.observability.prompt_id import SITE_ASSETS, fingerprint
from noctis.observability.prompt_ratchet import (
    CHANGELOG_PATH,
    RECORD_PATH,
    REGENERATE_COMMAND,
    build_record,
    check,
    compare_records,
    load_record,
    main,
    newest_entry,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET_SOURCE = REPO_ROOT / "src" / "noctis" / "observability" / "prompt_ratchet.py"

AUTHOR_FILE = "src/noctis/research/author.py"
AUTHOR_SIBLING = "src/noctis/research/contract_sheet.py"
IDEATION_FILE = "src/noctis/research/ideation.py"

BASELINE_ENTRY = (
    "## 2026-01-01 — sites: author, briefings, conversation, distill, episodic, ideation\n"
    "\nBaseline.\n"
)


def _changelog(*entries: str) -> str:
    """A changelog with the newest entry first, the way a reader reads it."""
    return "# Prompt changelog\n\nHow to read this file.\n\n" + "\n".join(entries)


def _build_tree(root: Path, *, changelog: str | None = None) -> Path:
    """A miniature repo: every prompt asset, a changelog, and files outside the map."""
    for rel in {rel for paths in SITE_ASSETS.values() for rel in paths}:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {rel}\nbody of {rel}\n")
    (root / CHANGELOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / CHANGELOG_PATH).write_text(
        _changelog(BASELINE_ENTRY) if changelog is None else changelog
    )
    (root / "docs" / "research.md").write_text("# research\n")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_prompt_ratchet.py").write_text("# a test\n")
    return root


def _declare(root: Path, heading_sites: str, note: str = "Reworded.") -> None:
    """Push a new entry onto the front of the changelog, naming the sites it explains."""
    existing = (root / CHANGELOG_PATH).read_text()
    body = existing.split("\n## ", 1)
    tail = "\n## " + body[1] if len(body) > 1 else ""
    (root / CHANGELOG_PATH).write_text(
        f"# Prompt changelog\n\nHow to read this file.\n\n"
        f"## 2026-02-02 — sites: {heading_sites}\n\n{note}\n{tail}"
    )


def _edit(root: Path, rel: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + '\nPROMPT = "a reworded prompt"\n')


# ── the changelog reader, pure ────────────────────────────────────────────────────────────


def test_the_newest_entry_is_the_first_one_and_names_its_sites():
    text = _changelog(
        "## 2026-02-02 — sites: author, ideation\n\nSharpened the brief.\n",
        "## 2026-01-01 — sites: distill\n\nOlder.\n",
    )

    entry = newest_entry(text)

    assert entry is not None
    assert entry.sites == ("author", "ideation")
    assert "2026-02-02" in entry.heading


def test_an_entry_with_no_sites_marker_declares_nothing():
    """Prose is not a declaration — the sites have to be named where a machine can read them."""
    entry = newest_entry(_changelog("## 2026-02-02\n\nReworded the author prompt a bit.\n"))

    assert entry is not None
    assert entry.sites == ()


def test_a_heading_inside_a_fenced_code_block_is_not_an_entry():
    """The page documents its own format in a fence — that template must declare nothing."""
    text = (
        "# Prompt changelog\n\n"
        "```text\n## <YYYY-MM-DD> — sites: <site>[, <site>…]\n```\n\n"
        "## 2026-02-02 — sites: author\n\nThe real entry.\n"
    )

    entry = newest_entry(text)

    assert entry is not None
    assert entry.sites == ("author",)
    assert "2026-02-02" in entry.heading


def test_a_fenced_block_inside_an_entry_does_not_end_it():
    """Prose after the fence still belongs to the entry, so amending it re-declares the change."""

    def entry_for(note: str):
        return newest_entry(
            _changelog(
                f"## 2026-02-02 — sites: author\n\n```text\n## not an entry\n```\n\n{note}\n",
                "## 2026-01-01 — sites: distill\n\nOlder.\n",
            )
        )

    before, after = entry_for("One change."), entry_for("One change, amended.")

    assert before is not None and after is not None
    assert before.sites == ("author",)
    assert before.digest != after.digest


def test_a_changelog_with_no_entries_at_all_reads_as_none():
    assert newest_entry("# Prompt changelog\n\nNothing yet.\n") is None


def test_editing_an_entrys_body_changes_its_identity():
    """Amending the newest entry is how a second change in one PR gets declared."""
    before = newest_entry(_changelog("## 2026-02-02 — sites: author\n\nOne change.\n"))
    after = newest_entry(_changelog("## 2026-02-02 — sites: author\n\nOne change, and another.\n"))

    assert before is not None and after is not None
    assert before.digest != after.digest


# ── the committed record ──────────────────────────────────────────────────────────────────


def test_a_record_holds_every_site_digest_and_the_changelog_head(tmp_path):
    record = build_record(_build_tree(tmp_path))

    assert set(record["sites"]) == set(SITE_ASSETS)
    assert record["sites"]["author"]["digest"] == fingerprint(tmp_path).digest("author")
    # Per-file digests, so a drift report names the file that moved, not its siblings.
    assert set(record["sites"]["author"]["files"]) == set(SITE_ASSETS["author"])
    assert record["changelog"]["path"] == CHANGELOG_PATH
    assert "2026-01-01" in record["changelog"]["heading"]
    assert record["regenerate_with"] == REGENERATE_COMMAND


def test_an_unchanged_tree_checks_ok(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)

    result = check(root)

    assert result.status == "ok"
    assert result.ok
    assert not result.drift


def test_editing_docs_or_a_test_never_fires_the_check(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    (root / "docs" / "research.md").write_text("# research, rewritten\n")
    (root / "tests" / "test_prompt_ratchet.py").write_text("# a new test\n")

    assert check(root).status == "ok"


# ── undeclared drift fails ────────────────────────────────────────────────────────────────


def test_drift_with_no_changelog_entry_fails_naming_the_site_and_the_file(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)

    result = check(root)

    assert result.status == "fail"
    assert not result.ok
    assert result.undeclared_drift
    assert [drift.site for drift in result.drift] == ["author"]
    assert result.drift[0].files == (AUTHOR_FILE,)
    report = result.report()
    assert "author" in report and AUTHOR_FILE in report
    assert AUTHOR_SIBLING not in report  # only what moved, never its untouched siblings
    assert CHANGELOG_PATH in report


def test_the_report_names_the_changelog_entry_it_read(tmp_path):
    """ "I wrote an entry and it still fails" needs the entry the check actually saw."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)

    report = check(root).report()

    assert "2026-01-01" in report  # the baseline entry, which declares nothing new
    assert CHANGELOG_PATH in report


def test_the_report_says_so_when_the_changelog_has_no_entry_at_all(tmp_path):
    root = _build_tree(tmp_path, changelog="# Prompt changelog\n\nNothing yet.\n")
    write_record(root)
    _edit(root, AUTHOR_FILE)

    report = check(root).report()

    assert "none" in report.lower()
    assert CHANGELOG_PATH in report


def test_an_entry_that_names_another_site_does_not_declare_this_one(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)
    _declare(root, "ideation")

    result = check(root)

    assert result.status == "fail"
    assert result.undeclared_drift
    assert "author" in result.report()


def test_an_entry_that_was_already_the_head_when_the_record_was_written_declares_nothing(tmp_path):
    """Yesterday's entry cannot license today's edit, or the ratchet is a standing permission."""
    root = _build_tree(tmp_path)
    _declare(root, "author")
    write_record(root)  # the record now names that entry as the head it was written against
    _edit(root, AUTHOR_FILE)

    result = check(root)

    assert result.status == "fail"
    assert result.undeclared_drift
    assert "changelog" in result.report().lower()


# ── a declared change lands ───────────────────────────────────────────────────────────────


def test_a_new_entry_naming_the_drifted_site_makes_the_drift_declared(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)
    _declare(root, "author")

    result = check(root)

    assert result.status == "fail"  # the record is still stale...
    assert not result.undeclared_drift  # ...but the change is declared, so --write may record it
    assert REGENERATE_COMMAND in result.report()


def test_regenerating_a_declared_change_writes_the_record_the_check_then_accepts(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)
    _declare(root, "author")

    assert main(["--write", "--root", str(root)]) == 0

    assert load_record(root)["sites"]["author"]["digest"] == fingerprint(root).digest("author")
    assert check(root).status == "ok"


def test_one_entry_may_declare_several_sites_at_once(tmp_path):
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)
    _edit(root, IDEATION_FILE)
    _declare(root, "author, ideation")

    result = check(root)

    assert [drift.site for drift in result.drift] == ["author", "ideation"]
    assert not result.undeclared_drift
    assert main(["--write", "--root", str(root)]) == 0
    assert check(root).status == "ok"


def test_a_second_undeclared_site_still_blocks_a_declared_one(tmp_path):
    """The strictest drift wins the verdict; the declared one is still named."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)
    _edit(root, IDEATION_FILE)
    _declare(root, "author")

    result = check(root)

    assert result.undeclared_drift
    assert [drift.site for drift in result.drift] == ["author", "ideation"]
    assert main(["--write", "--root", str(root)]) == 1


# ── --write refuses undeclared drift ──────────────────────────────────────────────────────


def test_regenerating_undeclared_drift_refuses_and_writes_nothing(tmp_path, capsys):
    """``--write`` is the fix every failure recommends, so it cannot also be how an undeclared
    prompt change gets recorded — the ratchet would reduce to whether the contributor read the
    failure before typing the command it printed."""
    root = _build_tree(tmp_path)
    write_record(root)
    before = (root / RECORD_PATH).read_bytes()
    _edit(root, AUTHOR_FILE)
    capsys.readouterr()

    code = main(["--write", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 1
    assert (root / RECORD_PATH).read_bytes() == before  # byte-identical: nothing was written
    assert "author" in out and AUTHOR_FILE in out
    assert CHANGELOG_PATH in out
    assert "refusing to regenerate" in out


def test_a_refused_regeneration_leaves_the_check_still_failing(tmp_path, capsys):
    """ "edit, regenerate, commit, CI passes" is not a reachable sequence."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)

    assert main(["--write", "--root", str(root)]) == 1
    assert main(["--check", "--root", str(root)]) == 1
    assert "author" in capsys.readouterr().out


def test_a_missing_record_still_regenerates_as_the_baseline(tmp_path):
    """There is nothing to compare against, and this is how the baseline is created."""
    root = _build_tree(tmp_path)
    assert load_record(root) is None

    assert main(["--write", "--root", str(root)]) == 0

    assert check(root).status == "ok"


def test_regenerating_an_unchanged_tree_is_idempotent(tmp_path):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    before = (root / RECORD_PATH).read_bytes()

    assert main(["--write", "--root", str(root)]) == 0

    assert (root / RECORD_PATH).read_bytes() == before


def test_the_pure_writer_stays_available_for_creating_a_baseline(tmp_path):
    """``write_record`` is the unguarded writer a fresh tree needs; the refusal is a decision in
    the ``--write`` path, so the comparison itself stays pure and unchanged."""
    root = _build_tree(tmp_path)
    write_record(root)
    _edit(root, AUTHOR_FILE)

    assert check(root).status == "fail"
    assert write_record(root) == root / RECORD_PATH
    assert check(root).status == "ok"


# ── a missing or malformed baseline ───────────────────────────────────────────────────────


def test_a_missing_record_file_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)

    result = check(root)

    assert result.status == "fail"
    assert load_record(root) is None
    assert RECORD_PATH in result.report()
    assert REGENERATE_COMMAND in result.report()


def test_a_malformed_record_file_fails_with_a_regenerate_message(tmp_path):
    root = _build_tree(tmp_path)
    (root / RECORD_PATH).write_text("{ not json at all")

    result = check(root)

    assert result.status == "fail"
    assert REGENERATE_COMMAND in result.report()


def test_a_site_that_is_null_on_both_sides_is_not_drift(tmp_path):
    root = _build_tree(tmp_path)
    (root / IDEATION_FILE).unlink()
    committed = build_record(root)

    result = compare_records(build_record(root), committed)

    assert committed["sites"]["ideation"]["digest"] is None
    assert result.status == "ok"


# ── the entrypoint CI runs ────────────────────────────────────────────────────────────────


def test_the_check_is_the_default_action(tmp_path):
    root = _build_tree(tmp_path)

    assert main(["--root", str(root)]) == 1  # no record committed yet
    main(["--write", "--root", str(root)])
    assert main(["--root", str(root)]) == 0


def test_the_check_prints_the_site_and_the_file_that_moved(tmp_path, capsys):
    root = _build_tree(tmp_path)
    main(["--write", "--root", str(root)])
    _edit(root, IDEATION_FILE)

    code = main(["--check", "--root", str(root)])

    out = capsys.readouterr().out
    assert code == 1
    assert "ideation" in out and IDEATION_FILE in out
    assert RECORD_PATH in out


# ── this checkout ─────────────────────────────────────────────────────────────────────────


def test_the_committed_prompt_fingerprint_file_records_this_checkout():
    """The file in the repo is the baseline CI compares against — it must be in sync now."""
    record = load_record(REPO_ROOT)

    assert record is not None
    fp = fingerprint(REPO_ROOT)
    for site in SITE_ASSETS:
        assert record["sites"][site]["digest"] == fp.digest(site), site
    assert check(REPO_ROOT).ok, check(REPO_ROOT).report()


def _committed_entry_sites() -> set[str]:
    """Every site any entry in the committed changelog names, read newest-first with the module's
    own entry parser (each pass consumes the heading it just read)."""
    text = (REPO_ROOT / CHANGELOG_PATH).read_text()
    sites: set[str] = set()
    while (entry := newest_entry(text)) is not None:
        sites.update(entry.sites)
        heading = f"## {entry.heading}"
        text = text[text.index(heading) + len(heading) :]
    return sites


def test_every_committed_site_hash_reads_back_to_a_changelog_entry():
    """The page's own promise: every hash in the record reads back to an entry here. The baseline
    entry names them all; each later entry names only the sites that moved with it."""
    assert _committed_entry_sites() >= set(SITE_ASSETS)


def test_the_newest_changelog_entry_names_only_sites_that_exist():
    """A declaration a machine cannot resolve to a site declares nothing — a typo'd site name
    would let real drift through under a heading that reads as if it covered it."""
    entry = newest_entry((REPO_ROOT / CHANGELOG_PATH).read_text())

    assert entry is not None
    assert set(entry.sites) <= set(SITE_ASSETS)


def test_the_prompt_record_is_separate_from_the_engine_record():
    """Prompts and arbiter behaviour drift on different clocks — two files, two clocks."""
    assert RECORD_PATH != "engine_fingerprint.json"
    assert (REPO_ROOT / "engine_fingerprint.json").is_file()
    assert "engine_version" not in json.loads((REPO_ROOT / RECORD_PATH).read_text())


# ── wiring and documentation ──────────────────────────────────────────────────────────────


def test_the_ratchet_runs_in_ci_and_in_pre_commit():
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    hooks = (REPO_ROOT / ".pre-commit-config.yaml").read_text()

    assert "scripts/prompt_fingerprint.py" in workflow
    assert "--check" in workflow
    assert "scripts/prompt_fingerprint.py" in hooks


def test_the_regeneration_command_is_documented():
    development = (REPO_ROOT / "docs" / "development.md").read_text()
    agents = (REPO_ROOT / "AGENTS.md").read_text()

    assert REGENERATE_COMMAND in development
    assert RECORD_PATH in development
    assert CHANGELOG_PATH in development
    assert "scripts/prompt_fingerprint.py" in agents


def test_the_declared_change_rule_is_documented_where_a_contributor_reads_it():
    development = (REPO_ROOT / "docs" / "development.md").read_text()
    script = (REPO_ROOT / "scripts" / "prompt_fingerprint.py").read_text()
    docstring = RATCHET_SOURCE.read_text().split('"""')[1].lower()

    assert "refuses to regenerate" in development
    assert "refuses" in script
    assert "changelog" in docstring and "declar" in docstring
    assert "--write" in docstring and "refus" in docstring


@pytest.mark.parametrize("site", sorted(SITE_ASSETS))
def test_every_site_is_named_on_the_changelog_page(site):
    """A hash has to read back to a human explanation, which starts with knowing the sites."""
    assert site in (REPO_ROOT / CHANGELOG_PATH).read_text()
