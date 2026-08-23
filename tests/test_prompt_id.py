"""Prompt-asset identity (#183): one content hash per LLM call site.

Every assertion here is external — a digest value, *which* sites moved under an edit, what the
pure per-site function returns for a tree. The root is injectable, so a scenario is a miniature
repo in ``tmp_path`` with exactly one file edited, exactly as ``tests/test_engine_id.py`` does.

The per-site split is the whole contract: prompt assets exist so "did the coder's brief move?"
is answerable separately from "was the distiller reworded?", and a benchmark record can name the
one site's prompt that produced a result. So independence gets a test per site, and the one
shared asset every site's assembled text renders — ``digests.py`` — gets its own, because
over-partitioning is the honest direction and silence is the failure this module exists to end.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from noctis.observability import engine_id
from noctis.observability.prompt_id import (
    SITE_ASSETS,
    compare,
    fingerprint,
    site_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ID_SOURCE = REPO_ROOT / "src" / "noctis" / "observability" / "prompt_id.py"

SHARED_ASSET = "src/noctis/research/digests.py"

# One asset per site that no other site lists — what per-site independence is asserted with.
EXCLUSIVE_ASSETS = {
    "author": "src/noctis/research/author.py",
    "briefings": "src/noctis/research/briefings.py",
    "conversation": "src/noctis/research/prompt.py",
    "distill": "src/noctis/research/distill.py",
    "episodic": "src/noctis/research/driver.py",
    "ideation": "src/noctis/research/ideation.py",
}


def _build_tree(root: Path) -> Path:
    """A miniature repo holding every path the site map names, plus files outside it."""
    for rel in {rel for paths in SITE_ASSETS.values() for rel in paths}:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {rel}\nbody of {rel}\n")
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "research.md").write_text("# research\n")
    (root / "src" / "noctis" / "research" / "tools.py").write_text("# tools\n")
    return root


def _edit(root: Path, rel: str) -> None:
    path = root / rel
    path.write_text(path.read_text() + '\nPROMPT = "a reworded prompt"\n')


# ── the map ───────────────────────────────────────────────────────────────────────────────


def test_every_listed_asset_exists_in_this_checkout():
    """A site whose file was renamed away fingerprints null — the map is the contract."""
    missing = [
        rel for paths in SITE_ASSETS.values() for rel in paths if not (REPO_ROOT / rel).is_file()
    ]

    assert missing == []


def test_every_site_has_an_asset_no_other_site_lists():
    """Independence is only meaningful if each site owns text of its own."""
    for site, rel in EXCLUSIVE_ASSETS.items():
        owners = [name for name, paths in SITE_ASSETS.items() if rel in paths]
        assert owners == [site], (site, owners)
    assert set(EXCLUSIVE_ASSETS) == set(SITE_ASSETS)


# ── stability ─────────────────────────────────────────────────────────────────────────────


def test_the_fingerprint_is_stable_when_nothing_moves(tmp_path):
    root = _build_tree(tmp_path)

    assert fingerprint(root).digests() == fingerprint(root).digests()
    assert compare(fingerprint(root).digests(), fingerprint(root).digests()) == set()


def test_editing_a_file_outside_the_map_moves_no_site(tmp_path):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    (root / "docs" / "research.md").write_text("# research, rewritten\n")
    (root / "src" / "noctis" / "research" / "tools.py").write_text("# tools, rewritten\n")

    assert compare(before.digests(), fingerprint(root).digests()) == set()


def test_a_crlf_checkout_fingerprints_identically(tmp_path):
    """The same normalization the engine fingerprint takes, so a digest is platform-independent."""
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    for rel in {rel for paths in SITE_ASSETS.values() for rel in paths}:
        path = root / rel
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert compare(before.digests(), fingerprint(root).digests()) == set()


# ── per-site independence ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("site", "rel"), sorted(EXCLUSIVE_ASSETS.items()))
def test_editing_one_sites_asset_moves_that_site_and_no_other(tmp_path, site, rel):
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    _edit(root, rel)

    assert compare(before.digests(), fingerprint(root).digests()) == {site}


def test_a_shared_asset_moves_every_site_that_assembles_it(tmp_path):
    """``digests.py`` renders facts four sites' prompts embed, so its edit moves all four.

    Over-partitioning is the accepted direction: a site whose assembled text changed must never
    keep its old identity, even when the words were written in a module it merely imports.
    """
    root = _build_tree(tmp_path)
    before = fingerprint(root)
    _edit(root, SHARED_ASSET)

    moved = compare(before.digests(), fingerprint(root).digests())
    assert moved == {name for name, paths in SITE_ASSETS.items() if SHARED_ASSET in paths}
    assert len(moved) > 1


# ── the pure per-site function ────────────────────────────────────────────────────────────


def test_the_per_site_hash_is_a_pure_function_of_the_tree(tmp_path):
    """What a future benchmark record's comparable key reads: site name in, content hash out."""
    root = _build_tree(tmp_path)

    for site in SITE_ASSETS:
        digest = site_digest(site, root)
        assert isinstance(digest, str) and digest
        assert digest == site_digest(site, root)
        assert digest == fingerprint(root).digest(site)

    assert len({site_digest(site, root) for site in SITE_ASSETS}) == len(SITE_ASSETS)


def test_the_per_site_hash_reads_this_checkout_by_default():
    assert site_digest("author") == fingerprint(REPO_ROOT).digest("author")


def test_an_unknown_site_is_refused_rather_than_answered_null():
    """A typo'd site name must not read as "this site has no prompt"."""
    with pytest.raises(KeyError):
        site_digest("coder")


# ── the missing-input rule ────────────────────────────────────────────────────────────────


def test_a_site_with_a_missing_asset_is_null_with_a_note(tmp_path):
    """A partial digest would look like an identity while meaning "some of the prompt"."""
    root = _build_tree(tmp_path)
    (root / EXCLUSIVE_ASSETS["distill"]).unlink()

    fp = fingerprint(root)

    assert fp.digest("distill") is None
    assert EXCLUSIVE_ASSETS["distill"] in (fp.note("distill") or "")
    assert fp.digest("ideation") is not None  # its neighbours are unharmed


def test_two_null_sites_are_not_a_difference(tmp_path):
    root = _build_tree(tmp_path)
    (root / EXCLUSIVE_ASSETS["distill"]).unlink()

    assert compare(fingerprint(root).digests(), fingerprint(root).digests()) == set()


# ── one hashing rule for the repo ─────────────────────────────────────────────────────────


def test_the_prompt_digest_uses_the_one_file_digest_the_engine_fingerprint_uses():
    """Two normalizations of "the content of a committed file" would eventually disagree."""
    from noctis.observability import prompt_id

    assert prompt_id.file_digest is engine_id.file_digest
    assert prompt_id.default_root is engine_id.default_root


def test_the_null_rule_is_the_engine_fingerprints_one_not_a_second_implementation():
    """Two answers to "did this digest move?" would eventually disagree, silently."""
    from noctis.observability import prompt_id

    assert prompt_id.compare is engine_id.compare
    assert "compare" in prompt_id.__all__


def test_the_prompt_module_defines_no_digest_projection_of_its_own():
    """The site map and the hashing are this module's job; reading a digest map is not."""
    defined = {
        node.name
        for node in ast.walk(ast.parse(PROMPT_ID_SOURCE.read_text()))
        if isinstance(node, ast.FunctionDef)
    }

    assert "compare" not in defined
    assert [name for name in defined if "digest_of" in name] == []


def test_the_prompt_fingerprint_is_a_separate_artifact_from_the_engine_one():
    """Prompts and arbiter behaviour drift on different clocks; neither key may bind the other."""
    source = PROMPT_ID_SOURCE.read_text()

    assert "ENGINE_VERSION" not in source
    assert "ARBITER_COMPONENTS" not in source
