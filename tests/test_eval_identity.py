"""Per-site identity (#193): a declared version paired with that site's prompt-asset hash.

The accessor is the one bridge between the site registry and the prompt-fingerprint module, so
what is asserted here is exactly what a record builder folding a ``comparable_key`` depends on:
the version a declaration carries, the hash the site's prompt assets currently compute to, which
sites move together, and the refusal when a site nobody declares is asked for.

Hash behaviour piggybacks on #168's technique — a miniature repo in ``tmp_path`` holding every
path the prompt-asset map names, with exactly one file edited — because the identity's hash is a
pure function of a tree, and the honest way to assert "it moves when the prompt moves" is to move
a prompt.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from noctis.eval.guard import eval_layer_importers
from noctis.eval.identity import (
    PROMPT_ASSET_GROUPS,
    SiteIdentity,
    site_identities,
    site_identity,
)
from noctis.eval.knobs import SiteKnobs
from noctis.eval.registry import SITES, UnknownSite, index_sites, sites
from noctis.eval.site import AgentSite
from noctis.observability.prompt_id import SITE_ASSETS, site_digest

SHARED_ASSET = "src/noctis/research/digests.py"

# One prompt asset per declared site that the site's mapped groups list — what "this site's own
# prompt moved" is asserted with. The three episodic sites share both of theirs, by design.
SITE_ASSET_EXAMPLES = [
    ("coder", "src/noctis/research/author.py"),
    ("coder", "src/noctis/research/contract_sheet.py"),
    ("formulate", "src/noctis/research/briefings.py"),
    ("formulate", "src/noctis/research/driver.py"),
    ("decide", "src/noctis/research/briefings.py"),
    ("decide", "src/noctis/research/driver.py"),
    ("discover", "src/noctis/research/briefings.py"),
    ("discover", "src/noctis/research/driver.py"),
    ("distill", "src/noctis/research/distill.py"),
]

EPISODIC_SITE_IDS = ("formulate", "decide", "discover")


def _build_tree(root: Path) -> Path:
    """A miniature repo holding every path the prompt-asset map names, plus files outside it."""
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


def _hashes(root: Path) -> dict[str, str | None]:
    return {identity.site_id: identity.prompt_asset_hash for identity in site_identities(root)}


def _scratch_site(site_id: str) -> AgentSite[str, str]:
    return AgentSite(
        id=site_id,
        version="1",
        contract=None,
        render=lambda site_input, spec: site_input,
        knobs=SiteKnobs,
    )


# ── the declared version ──────────────────────────────────────────────────────────────────
def test_every_declared_site_carries_a_non_empty_version_string() -> None:
    """The hand-bumped contract generation — a record can refuse a comparison across it."""
    for declaration in sites():
        assert isinstance(declaration.version, str)
        assert declaration.version.strip()


def test_a_sites_identity_reports_the_version_its_own_declaration_carries() -> None:
    for site_id, declaration in SITES.items():
        assert site_identity(site_id).version == declaration.version


# ── the eval-site → prompt-asset-group map ────────────────────────────────────────────────
def test_every_declared_site_maps_to_prompt_asset_groups_the_fingerprint_module_declares() -> None:
    assert set(PROMPT_ASSET_GROUPS) == set(SITES)
    for site_id, groups in PROMPT_ASSET_GROUPS.items():
        assert groups, site_id
        assert set(groups) <= set(SITE_ASSETS), site_id


def test_a_site_identity_names_the_prompt_asset_groups_its_hash_was_taken_over() -> None:
    """The hash is traceable: a reader can recompute it from the groups the identity names."""
    identity = site_identity("coder")

    assert identity.prompt_asset_groups == PROMPT_ASSET_GROUPS["coder"]


# ── the accessor ──────────────────────────────────────────────────────────────────────────
def test_every_declared_site_resolves_to_an_identity_in_registry_order() -> None:
    assert [identity.site_id for identity in site_identities()] == [
        declaration.id for declaration in sites()
    ]


def test_looking_up_an_identity_for_a_site_nothing_declares_names_the_declared_ids() -> None:
    with pytest.raises(UnknownSite) as error:
        site_identity("optimizer")

    message = str(error.value)
    assert "optimizer" in message
    assert "coder" in message
    assert "distill" in message


def test_a_declared_site_with_no_prompt_asset_row_is_refused_rather_than_hashed() -> None:
    """A new declaration must arrive with its prompt assets named, not with a silent null hash."""
    scratch = index_sites([_scratch_site("smuggled")])

    with pytest.raises(ValueError) as error:
        site_identity("smuggled", registry=scratch)

    assert "smuggled" in str(error.value)


def test_a_site_identity_cannot_be_reassigned_after_construction() -> None:
    identity = site_identity("distill")

    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.version = "2"  # type: ignore[misc]


def test_an_identity_reads_back_as_the_plain_data_a_record_builder_folds() -> None:
    identity = site_identity("distill")

    assert isinstance(identity, SiteIdentity)
    assert identity.site_id == "distill"
    assert isinstance(identity.prompt_asset_hash, str)
    assert identity.prompt_asset_hash


# ── the hash: this checkout ───────────────────────────────────────────────────────────────
def test_a_single_group_sites_hash_is_the_fingerprint_modules_own_digest_for_that_group() -> None:
    """Same rule, same file set, so the coder's identity is greppable in the committed record."""
    assert site_identity("coder").prompt_asset_hash == site_digest("author")
    assert site_identity("distill").prompt_asset_hash == site_digest("distill")


def test_the_identity_accessor_reads_this_checkout_when_no_root_is_given() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert site_identity("formulate") == site_identity("formulate", repo_root)


# ── the hash: what moves it, and what does not ────────────────────────────────────────────
def test_the_prompt_asset_hash_is_stable_when_no_prompt_asset_moves(tmp_path: Path) -> None:
    root = _build_tree(tmp_path)

    assert _hashes(root) == _hashes(root)


def test_editing_a_file_no_site_lists_moves_no_identity(tmp_path: Path) -> None:
    root = _build_tree(tmp_path)
    before = _hashes(root)
    (root / "docs" / "research.md").write_text("# research, rewritten\n")
    (root / "src" / "noctis" / "research" / "tools.py").write_text("# tools, rewritten\n")

    assert _hashes(root) == before


@pytest.mark.parametrize(("site_id", "rel"), SITE_ASSET_EXAMPLES)
def test_editing_a_sites_prompt_asset_changes_that_sites_hash(
    tmp_path: Path, site_id: str, rel: str
) -> None:
    root = _build_tree(tmp_path)
    before = site_identity(site_id, root).prompt_asset_hash
    _edit(root, rel)

    assert site_identity(site_id, root).prompt_asset_hash != before


def test_rewording_the_distillers_prompt_leaves_the_coders_hash_unmoved(tmp_path: Path) -> None:
    """Per-site identity is the point: an unrelated site's reword must not invalidate a result."""
    root = _build_tree(tmp_path)
    before = site_identity("coder", root).prompt_asset_hash
    _edit(root, "src/noctis/research/distill.py")

    assert site_identity("coder", root).prompt_asset_hash == before


def test_rewording_an_episodic_stage_leaves_the_distillers_hash_unmoved(tmp_path: Path) -> None:
    root = _build_tree(tmp_path)
    before = site_identity("distill", root).prompt_asset_hash
    _edit(root, "src/noctis/research/driver.py")

    assert site_identity("distill", root).prompt_asset_hash == before


def test_the_three_episodic_sites_share_one_prompt_asset_hash(tmp_path: Path) -> None:
    """They are assembled from the same prompt-bearing modules, and the hash says so honestly."""
    root = _build_tree(tmp_path)

    shared = {site_identity(site_id, root).prompt_asset_hash for site_id in EPISODIC_SITE_IDS}

    assert len(shared) == 1
    assert shared != {None}


def test_editing_the_shared_fact_renderer_moves_every_site_that_assembles_it(
    tmp_path: Path,
) -> None:
    """Over-partitioning is the accepted direction: assembled text moved, so identity moves."""
    root = _build_tree(tmp_path)
    before = _hashes(root)
    _edit(root, SHARED_ASSET)

    moved = {site_id for site_id, digest in _hashes(root).items() if digest != before[site_id]}
    assert moved == {"coder", *EPISODIC_SITE_IDS}


def test_a_site_whose_prompt_asset_is_missing_has_no_hash_rather_than_a_partial_one(
    tmp_path: Path,
) -> None:
    """A partial digest would look like an identity while meaning "some of the prompt"."""
    root = _build_tree(tmp_path)
    (root / "src" / "noctis" / "research" / "briefings.py").unlink()

    assert site_identity("formulate", root).prompt_asset_hash is None
    assert site_identity("distill", root).prompt_asset_hash is not None


# ── the one-way boundary ──────────────────────────────────────────────────────────────────
def test_no_engine_module_imports_the_identity_accessor() -> None:
    """The bridge crosses eval → engine only; the engine may never read a site's identity."""
    reaches = [violation.line() for violation in eval_layer_importers()]

    assert reaches == []
