"""The shared changelog reader (#354): one parser, one clause grammar, for both ratchets.

A ratchet's record says what the tree *is*; a changelog entry is the human sentence a moved digest
reads back to. How an entry is read is written once in :mod:`noctis.observability.changelog` — so
it is tested once here, over text, with no policy in sight: nothing in this file knows what a site
or a component is, exactly as the module does not.

Every assertion is external behaviour — the entry a text parses to, the names a clause reads back,
the block a record stores, the line a report prints. The one file this module touches is the page
itself, so a scenario is a string; the one I/O reading gets a ``tmp_path``.

The last test is the move's invariant: every heading already committed in
``docs/prompt-changelog.md`` declares the same sites under the clause grammar as it did under the
prompt policy's old "everything after ``sites:``" reading — which is what makes
``prompt_fingerprint.json`` byte-identical across the move.
"""

from __future__ import annotations

from pathlib import Path

from noctis.observability.changelog import (
    ChangelogEntry,
    declared_since,
    footer,
    header,
    newest_entry,
    read_entry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT_CHANGELOG = "docs/prompt-changelog.md"
SITES = "sites"


def _page(*entries: str) -> str:
    """A changelog page with the newest entry first, the way a reader reads it."""
    return "# A changelog\n\nHow to read this file.\n\n" + "\n".join(entries)


def _entry(text: str) -> ChangelogEntry:
    """The newest entry of a page that has one — the tests below all do."""
    entry = newest_entry(text)
    assert entry is not None
    return entry


# ── which entry is the newest one ─────────────────────────────────────────────────────────


def test_the_newest_entry_is_the_first_one_on_the_page():
    """Newest-first is how the page is written, so nothing parses a date or trusts an ordering."""
    entry = _entry(
        _page(
            "## 2026-02-02 — sites: author, ideation\n\nSharpened the brief.\n",
            "## 2026-01-01 — sites: distill\n\nOlder.\n",
        )
    )

    assert isinstance(entry, ChangelogEntry)
    assert entry.heading == "2026-02-02 — sites: author, ideation"
    assert entry.names(SITES) == ("author", "ideation")


def test_a_heading_inside_a_fenced_code_block_is_not_an_entry():
    """A page documents its own format in a fence — that template must declare nothing."""
    entry = _entry(
        "# A changelog\n\n"
        "```text\n## <YYYY-MM-DD> — sites: <site>[, <site>…]\n```\n\n"
        "## 2026-02-02 — sites: author\n\nThe real entry.\n"
    )

    assert entry.heading == "2026-02-02 — sites: author"
    assert entry.names(SITES) == ("author",)


def test_a_fenced_block_inside_an_entry_does_not_end_it():
    """Prose after the fence still belongs to the entry, so amending it re-declares the change."""

    def entry_for(note: str) -> ChangelogEntry:
        return _entry(
            _page(
                f"## 2026-02-02 — sites: author\n\n```text\n## not an entry\n```\n\n{note}\n",
                "## 2026-01-01 — sites: distill\n\nOlder.\n",
            )
        )

    before, after = entry_for("One change."), entry_for("One change, amended.")

    assert before.names(SITES) == ("author",)
    assert before.digest != after.digest


def test_a_page_with_no_entries_at_all_reads_as_none():
    assert newest_entry("# A changelog\n\nNothing yet.\n") is None


def test_editing_an_entrys_body_changes_its_identity():
    """Amending the newest entry is how a second change in one PR gets declared."""
    before = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))
    after = _entry(_page("## 2026-02-02 — sites: author\n\nOne change, and another.\n"))

    assert before.heading == after.heading
    assert before.digest != after.digest


# ── the clause grammar ────────────────────────────────────────────────────────────────────


def test_several_clauses_on_one_heading_parse_independently():
    """The grammar one reader serves two policies with: each clause read on its own."""
    entry = _entry(
        _page("## 2026-08-25 — components: backtest, gates — behaviour: unchanged\n\nA no-op.\n")
    )

    assert entry.names("components") == ("backtest", "gates")
    assert entry.names("behaviour") == ("unchanged",)


def test_a_clause_key_matches_case_insensitively():
    """The key is the machine-readable half, so its spelling is not the declaration's meaning."""
    entry = _entry(_page("## 2026-02-02 — Sites: author\n\nOne change.\n"))

    assert entry.names("sites") == ("author",)
    assert entry.names("SITES") == ("author",)


def test_the_date_is_not_a_clause():
    """A heading opens with its date, which carries no colon and declares nothing."""
    entry = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))

    assert dict(entry.clauses) == {"sites": "author"}
    assert entry.names("2026-02-02") == ()


def test_a_heading_with_no_clauses_at_all_declares_nothing():
    """Prose is not a declaration — a name has to arrive where a machine can read it."""
    entry = _entry(_page("## 2026-02-02\n\nReworded the author prompt a bit.\n"))

    assert dict(entry.clauses) == {}
    assert entry.names(SITES) == ()


def test_names_on_a_clause_the_heading_does_not_carry_is_empty():
    entry = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))

    assert entry.names("components") == ()


# ── the page itself ───────────────────────────────────────────────────────────────────────


def test_the_newest_entry_of_a_page_is_read_from_the_tree(tmp_path):
    page = tmp_path / "docs" / "a-changelog.md"
    page.parent.mkdir(parents=True)
    page.write_text(_page("## 2026-02-02 — sites: author\n\nOne change.\n"), encoding="utf-8")

    entry = read_entry(tmp_path, "docs/a-changelog.md")

    assert entry is not None
    assert entry.names(SITES) == ("author",)


def test_a_page_that_does_not_exist_reads_as_no_entry(tmp_path):
    """A missing page is a tree with nothing declared, never an error."""
    assert read_entry(tmp_path, "docs/a-changelog.md") is None


# ── the block a record stores, and the reading over two records ───────────────────────────


def test_the_header_block_is_the_page_head_a_record_is_written_against():
    entry = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))

    block = header(PROMPT_CHANGELOG, entry, entry.names(SITES))["changelog"]

    assert block == {
        "path": PROMPT_CHANGELOG,
        "heading": "2026-02-02 — sites: author",
        "digest": entry.digest,
        "declares": ["author"],
    }


def test_a_page_with_no_entry_stores_a_head_of_nulls():
    block = header(PROMPT_CHANGELOG, None)["changelog"]

    assert block == {
        "path": PROMPT_CHANGELOG,
        "heading": None,
        "digest": None,
        "declares": [],
    }


def test_an_entry_that_was_already_the_head_declares_nothing_new():
    """The second half of every declared-change rule: a matching digest means the page gained
    nothing since the record, so the entry is a standing permission rather than a declaration."""
    entry = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))
    record = header(PROMPT_CHANGELOG, entry, entry.names(SITES))

    assert declared_since(record, record) == frozenset()


def test_a_newer_entry_declares_the_names_on_its_clause():
    older = _entry(_page("## 2026-01-01 — sites: distill\n\nOlder.\n"))
    newer = _entry(_page("## 2026-02-02 — sites: author, ideation\n\nNewer.\n"))

    declared = declared_since(
        header(PROMPT_CHANGELOG, newer, newer.names(SITES)),
        header(PROMPT_CHANGELOG, older, older.names(SITES)),
    )

    assert declared == frozenset({"author", "ideation"})


def test_a_committed_record_with_no_changelog_block_reads_as_no_head_at_all():
    """A record written before its policy stored a head declares nothing, so any entry is new
    against it — the baseline case a ratchet gaining this block has to survive."""
    entry = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))

    declared = declared_since(header(PROMPT_CHANGELOG, entry, entry.names(SITES)), {})

    assert declared == frozenset({"author"})


def test_a_computed_record_with_no_entry_declares_nothing():
    assert declared_since(header(PROMPT_CHANGELOG, None), {}) == frozenset()


# ── the footer line every report prints ───────────────────────────────────────────────────


def test_the_footer_names_the_entry_the_check_read():
    """ "I wrote one and it still fails" is the question these tools get asked."""
    entry = _entry(_page("## 2026-02-02 — sites: author\n\nOne change.\n"))

    assert footer(PROMPT_CHANGELOG, header(PROMPT_CHANGELOG, entry)) == (
        f"newest {PROMPT_CHANGELOG} entry: 2026-02-02 — sites: author",
    )


def test_the_footer_says_none_when_the_page_has_no_entry():
    """A fenced or mis-headed entry is invisible without this line."""
    assert footer(PROMPT_CHANGELOG, header(PROMPT_CHANGELOG, None)) == (
        f"newest {PROMPT_CHANGELOG} entry: none",
    )


# ── this checkout: the move's invariant ───────────────────────────────────────────────────


def _sites_after_the_marker(heading: str) -> tuple[str, ...]:
    """The prompt policy's reading *before* the clause grammar: everything after ``sites:``,
    comma-separated. Kept here as the old half of the equivalence the next test asserts."""
    marker = heading.lower().find(f"{SITES}:")
    if marker < 0:
        return ()
    listed = heading[marker + len(SITES) + 1 :]
    return tuple(name for name in (part.strip() for part in listed.split(",")) if name)


def _committed_headings() -> list[str]:
    """Every ``## `` heading on the committed prompt changelog, fenced template included."""
    text = (REPO_ROOT / PROMPT_CHANGELOG).read_text(encoding="utf-8")
    return [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]


def test_every_committed_heading_declares_the_same_sites_under_the_clause_grammar():
    """The move's invariant, and why ``prompt_fingerprint.json`` is byte-identical after it: no
    heading already on the page means anything different once the parser reads clauses."""
    headings = _committed_headings()

    assert len(headings) > 1  # the page has real entries; an empty list would assert nothing
    for heading in headings:
        entry = _entry(f"## {heading}\n\nA body.\n")
        assert entry.names(SITES) == _sites_after_the_marker(heading), heading
