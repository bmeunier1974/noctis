"""QA-area retention (epic #36, story #42): prune-on-start to ``qa.keep_last_runs``.

The behaviors under test are external — which entries survive on disk after a prune. A run
folder is identified strictly by the run-id name shape (``20260720T144233Z-a3f9c1``, a
directory), and recency is name order because the run id sorts chronologically. Everything
else in the QA area (notes, dotfiles, near-miss names, a stray file matching the shape) is
never touched — retention prunes *run folders* and nothing else.
"""

from __future__ import annotations

from noctis.observability.debug import prune_qa_dir


def _run_id(i: int) -> str:
    """A valid, chronologically-ordered run id: the minute field carries the ordering.

    ``i`` (0..59) drives both the minute field and the 6-hex suffix, so larger ``i`` sorts
    later (more recent) by plain name order and every id is distinct.
    """
    return f"20260720T14{i:02d}33Z-{i:06x}"


def test_prunes_to_newest_n_run_folders(tmp_path):
    """More than N run folders present → only the newest N (by run-id order) survive."""
    qa = tmp_path / "qa"
    qa.mkdir()
    ids = [_run_id(i) for i in range(25)]
    for rid in ids:
        (qa / rid).mkdir()

    pruned = prune_qa_dir(qa, keep=20)

    survivors = sorted(p.name for p in qa.iterdir())
    assert survivors == sorted(ids[-20:])  # the newest 20 by name order
    assert sorted(pruned) == sorted(ids[:5])  # the oldest 5 removed


def test_non_run_entries_are_left_untouched(tmp_path):
    """Files, plain/hidden dirs, and near-miss (wrong-shape) names are never pruned."""
    qa = tmp_path / "qa"
    qa.mkdir()
    for rid in (_run_id(i) for i in range(25)):
        (qa / rid).mkdir()
    (qa / "notes.md").write_text("keep me")
    (qa / "keep").mkdir()
    (qa / ".hidden").mkdir()
    (qa / "20260720T144233Z-XYZ123").mkdir()  # uppercase → not run-id hex
    (qa / "not-a-run").mkdir()

    prune_qa_dir(qa, keep=20)

    names = {p.name for p in qa.iterdir()}
    assert {"notes.md", "keep", ".hidden", "20260720T144233Z-XYZ123", "not-a-run"} <= names


def test_a_file_named_like_a_run_id_is_never_pruned(tmp_path):
    """Only directories are run folders; a stray file matching the shape stays put — even
    when its name is the oldest, so recency alone would have selected it for removal."""
    qa = tmp_path / "qa"
    qa.mkdir()
    for i in range(1, 21):  # 20 real run folders
        (qa / _run_id(i)).mkdir()
    decoy = qa / _run_id(0)  # oldest name, but a FILE not a folder
    decoy.write_text("not a run folder")

    pruned = prune_qa_dir(qa, keep=20)

    assert pruned == []  # 20 folders, keep 20 → nothing pruned; the file is not a folder
    assert decoy.exists() and decoy.is_file()


def test_missing_qa_dir_is_a_noop(tmp_path):
    """A never-created QA area prunes to nothing without raising."""
    assert prune_qa_dir(tmp_path / "does-not-exist", keep=20) == []


def test_fewer_than_keep_run_folders_prunes_nothing(tmp_path):
    """At or below the retention count, every run folder survives."""
    qa = tmp_path / "qa"
    qa.mkdir()
    ids = [_run_id(i) for i in range(5)]
    for rid in ids:
        (qa / rid).mkdir()

    assert prune_qa_dir(qa, keep=20) == []
    assert {p.name for p in qa.iterdir()} == set(ids)


def test_keep_zero_prunes_all_run_folders_but_spares_decoys(tmp_path):
    """``keep=0`` means keep nothing: every run folder goes, non-run entries stay."""
    qa = tmp_path / "qa"
    qa.mkdir()
    ids = [_run_id(i) for i in range(3)]
    for rid in ids:
        (qa / rid).mkdir()
    (qa / "notes.md").write_text("x")

    pruned = prune_qa_dir(qa, keep=0)

    assert sorted(pruned) == sorted(ids)
    assert {p.name for p in qa.iterdir()} == {"notes.md"}


def test_negative_keep_is_treated_as_zero(tmp_path):
    """A negative keep is nonsense input: it must not slice off the *oldest* few by accident —
    it prunes all run folders, exactly as ``keep=0`` does."""
    qa = tmp_path / "qa"
    qa.mkdir()
    ids = [_run_id(i) for i in range(4)]
    for rid in ids:
        (qa / rid).mkdir()

    pruned = prune_qa_dir(qa, keep=-2)

    assert sorted(pruned) == sorted(ids)  # all gone, not just the oldest 2
    assert list(qa.iterdir()) == []


def test_accepts_a_str_path(tmp_path):
    """The QA dir may arrive as a plain string (a config field), not only a Path."""
    qa = tmp_path / "qa"
    qa.mkdir()
    ids = [_run_id(i) for i in range(22)]
    for rid in ids:
        (qa / rid).mkdir()

    pruned = prune_qa_dir(str(qa), keep=20)

    assert sorted(pruned) == sorted(ids[:2])
