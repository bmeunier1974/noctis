"""``bench corpus --site`` (#220): validation, stratification stats, and the split balance.

The third bench verb, and the only one that answers a question about the *corpus* rather than about
a run over it: what is in it, does all of it still load, and how is it divided. Three properties
carry the story here.

* **Everything it counts, it validated first.** Every case file goes through the site's own real
  loader — the coder's bucket-walking provider and coder schema, the generic flat provider for
  everybody else — so a file that has rotted is a refusal naming the file and the defect, not a
  number quietly computed over nineteen of twenty cases.
* **The verb names no site.** Which loader serves which corpus, and which vocabulary its labels are
  declared against, is one lookup in the eval layer's own table
  (:data:`~noctis.eval.bootstrap.SITE_CORPORA`), so a flat second corpus reports through exactly the
  same code path with its own axes — and a site with no buckets says so rather than printing an
  empty table.
* **Reading a corpus writes nothing.** No stamping, no repair, no index: a corpus is evidence, and
  a read that mutates its subject is how evidence stops being evidence.

The rendering is refusal-first like ``bench report``: an absent figure is the literal ``n/a`` and
never a zero, a declared bucket or axis level nobody has written a case for is named as declared and
unrepresented, and a case whose file declares no ``split:`` is counted **unstamped** — dealt in
memory at load time, and therefore still able to move when the corpus grows.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from noctis.cli import app
from noctis.eval.bootstrap import cases_root, flat_provider, load_corpus, site_vocabulary
from noctis.eval.coder_case import AXIS_LEVELS, Axis, Bucket
from noctis.eval.coder_corpus import CODER_SITE_ID, CoderCaseProvider
from noctis.eval.corpus_report import CorpusVocabulary, read_corpus, render_corpus_report

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_CASES = REPO_ROOT / "cases"

# The easy end of every axis: a valid, coherent label set a fixture varies one axis of.
EASY_AXES: Mapping[str, str] = {axis.value: AXIS_LEVELS[axis][0] for axis in Axis}

BRIEF: Mapping[str, str] = {
    "thesis": "A fast moving average above a slow one says demand outpaces its own average.",
    "entry_exit": "Long while the fast SMA is above the slow SMA; flat otherwise. No shorting.",
    "param_space": "fast 3-30, slow 20-100",
    "scenarios": "trend_ride_then_rollover: long during the trend, flat after the selloff.",
}


# ── fixture corpora ──────────────────────────────────────────────────────────────────────────


def _coder_case(
    tmp_path,
    name: str,
    *,
    bucket: Bucket = Bucket.CANARY,
    split: str | None = "tuning",
    **axes: str,
) -> Path:
    """One coder case in ``<workspace>/cases/coder/<bucket>/<name>.yaml`` — the layout it walks."""
    document: dict[str, Any] = {
        "site_id": CODER_SITE_ID,
        "payload": dict(BRIEF),
        "provenance": "authored:2026-08-02",
        "tags": [f"bucket:{bucket.value}"],
        "difficulty": {**EASY_AXES, **axes},
    }
    if split is not None:
        document["split"] = split
    directory = cases_root(tmp_path / "workspace") / CODER_SITE_ID / bucket.value
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


def _decide_case(tmp_path, name: str, *, split: str | None = "tuning", **axes: str) -> Path:
    """One flat case for a second site — the generic layout every other corpus uses."""
    document: dict[str, Any] = {
        "site_id": "decide",
        "payload": {"candidate": name, "scorecard": {"holdout": 1.0}},
        "provenance": "authored:2026-08-02",
        "difficulty": dict(axes),
    }
    if split is not None:
        document["split"] = split
    directory = cases_root(tmp_path / "workspace") / "decide"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    return path


def _corpus(tmp_path) -> None:
    """A small coder corpus: four canaries and two edge cases, every split stamped."""
    _coder_case(tmp_path, "canary-one", split="tuning")
    _coder_case(tmp_path, "canary-two", split="tuning")
    _coder_case(tmp_path, "canary-three", split="holdout")
    _coder_case(tmp_path, "canary-four", split="tuning", api_surface="exits")
    _coder_case(tmp_path, "edge-one", bucket=Bucket.EDGE, split="tuning", api_surface="exits")
    _coder_case(tmp_path, "edge-two", bucket=Bucket.EDGE, split="holdout", api_surface="exits")


def _row(rendered: str, label: str) -> str:
    """The value of the one row carrying ``label`` — the report, read as an operator reads it."""
    rows = [line for line in rendered.splitlines() if line.strip().startswith(label)]
    assert rows, f"no row labelled {label!r} in:\n{rendered}"
    return rows[0].strip()[len(label) :].strip()


def _block(rendered: str, heading: str) -> str:
    """Everything under ``heading``, up to the next line indented no further than it."""
    lines = rendered.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == heading]
    assert starts, f"no block headed {heading!r} in:\n{rendered}"
    start = starts[0]
    depth = len(lines[start]) - len(lines[start].lstrip())
    kept: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= depth:
            break
        kept.append(line)
    return "\n".join(kept)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``root`` with its bytes and its mtime — a tree, pinned."""
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ── validation: everything it counts, it loaded ──────────────────────────────────────────────


def test_bench_corpus_states_how_many_cases_validated_and_the_digest_they_hash_to(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    assert result.exit_code == 0, result.output
    assert "6 case(s) validated" in result.output
    assert "digest" in result.output


def test_bench_corpus_refuses_a_malformed_case_file_naming_the_file_and_the_defect(tmp_path):
    _corpus(tmp_path)
    broken = cases_root(tmp_path / "workspace") / CODER_SITE_ID / "canary" / "rotted.yaml"
    broken.write_text(
        yaml.safe_dump(
            {
                "site_id": CODER_SITE_ID,
                "payload": dict(BRIEF),
                "provenance": "authored:2026-08-02",
                "tags": [f"bucket:{Bucket.CANARY.value}"],
                "difficulty": dict(EASY_AXES),
                "expected": "a strategy file",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    assert result.exit_code == 1
    assert str(broken) in result.output
    assert "'expected' is refused" in result.output
    assert "case(s) validated" not in result.output


def test_bench_corpus_refuses_an_absent_corpus_naming_the_directory_it_looked_in(tmp_path):
    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    assert result.exit_code == 1
    assert str(cases_root(tmp_path / "workspace") / CODER_SITE_ID) in result.output


def test_bench_corpus_refuses_a_site_nothing_declares_naming_the_ones_that_are(tmp_path):
    result = runner.invoke(app, ["bench", "corpus", "--site", "codr"])

    assert result.exit_code == 1
    assert "coder" in result.output


def test_the_site_lookup_hands_the_coder_corpus_to_every_reader_that_asks_for_it(tmp_path):
    """One table decides which reader serves a corpus, so `bench run` sees the same cases."""
    _corpus(tmp_path)
    root = cases_root(tmp_path / "workspace")

    assert len(load_corpus(CODER_SITE_ID, cases_root=root)) == 6
    assert flat_provider(root, None).load(CODER_SITE_ID) == ()


# ── stratification stats: buckets and axes ───────────────────────────────────────────────────


def test_bench_corpus_counts_the_cases_filed_in_every_bucket_of_the_site(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    buckets = _block(result.output, "Buckets")
    assert _row(buckets, "canary").startswith("4 case(s)")
    assert _row(buckets, "edge").startswith("2 case(s)")


def test_bench_corpus_names_a_bucket_the_site_declares_that_holds_no_case_yet(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    buckets = _block(result.output, "Buckets")
    assert _row(buckets, "field") == "0 case(s) — n/a (declared, no case represents it)"
    assert _row(buckets, "replay") == "0 case(s) — n/a (declared, no case represents it)"


def test_bench_corpus_counts_the_cases_at_every_level_the_axes_are_labelled_with(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    surface = _block(_block(result.output, "Axes"), Axis.API_SURFACE.value)
    assert _row(surface, "bars_only").startswith("3 case(s)")
    assert _row(surface, "exits").startswith("3 case(s)")


def test_bench_corpus_names_an_axis_level_the_site_declares_that_no_case_represents(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    surface = _block(_block(result.output, "Axes"), Axis.API_SURFACE.value)
    assert _row(surface, "indicators") == "0 case(s) — n/a (declared, no case represents it)"


# ── split balance: overall, per bucket, per axis level ───────────────────────────────────────


def test_bench_corpus_states_the_tuning_and_holdout_counts_and_shares_over_the_whole_corpus(
    tmp_path,
):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    balance = _block(result.output, "Split balance")
    assert _row(balance, "corpus") == (
        "6 case(s) — tuning 4 (0.6667), holdout 2 (0.3333), unstamped 0"
    )


def test_bench_corpus_states_the_split_balance_within_each_bucket(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    buckets = _block(result.output, "Buckets")
    assert _row(buckets, "canary").endswith("tuning 3 (0.7500), holdout 1 (0.2500), unstamped 0")
    assert _row(buckets, "edge").endswith("tuning 1 (0.5000), holdout 1 (0.5000), unstamped 0")


def test_bench_corpus_states_the_split_balance_within_each_axis_level(tmp_path):
    _corpus(tmp_path)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    surface = _block(_block(result.output, "Axes"), Axis.API_SURFACE.value)
    assert _row(surface, "bars_only").endswith("tuning 2 (0.6667), holdout 1 (0.3333), unstamped 0")


def test_bench_corpus_counts_a_case_whose_file_declares_no_split_as_unstamped(tmp_path):
    _corpus(tmp_path)
    _coder_case(tmp_path, "canary-unstamped", split=None, warmup_arithmetic="composed")

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    balance = _block(result.output, "Split balance")
    assert _row(balance, "corpus").endswith("unstamped 1")


# ── a second site, through the same code path ────────────────────────────────────────────────


def test_a_flat_second_sites_corpus_reports_its_own_axes_through_the_same_verb(tmp_path):
    _decide_case(tmp_path, "decide-one", margin="near", evidence_depth="at_floor")
    _decide_case(tmp_path, "decide-two", split="holdout", margin="comfortable")

    result = runner.invoke(app, ["bench", "corpus", "--site", "decide"])

    assert result.exit_code == 0, result.output
    axes = _block(result.output, "Axes")
    assert _row(_block(axes, "margin"), "near").startswith("1 case(s)")
    assert _row(_block(axes, "margin"), "comfortable").startswith("1 case(s)")
    assert _row(_block(axes, "evidence_depth"), "at_floor").startswith("1 case(s)")


def test_a_site_whose_cases_carry_no_buckets_says_so_rather_than_printing_an_empty_table(tmp_path):
    _decide_case(tmp_path, "decide-one", margin="near")

    result = runner.invoke(app, ["bench", "corpus", "--site", "decide"])

    assert result.exit_code == 0, result.output
    assert _block(result.output, "Buckets").strip() == "n/a — the decide corpus declares no buckets"


# ── the verb's wiring, and its read-only promise ─────────────────────────────────────────────


def test_the_bench_group_lists_the_corpus_verb_beside_run_and_report(tmp_path):
    result = runner.invoke(app, ["bench", "--help"])

    assert result.exit_code == 0, result.output
    assert "corpus" in result.output


def test_importing_the_engine_cli_still_loads_no_benchmark_code_with_the_corpus_verb_wired():
    """The deferred delegation, from the runtime side: a third verb buys no engine-side coupling."""
    probe = "import noctis.cli, sys; print([m for m in sys.modules if m.startswith('noctis.eval')])"

    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert loaded.stdout.strip() == "[]"


def test_reading_a_corpus_writes_nothing_into_the_cases_tree(tmp_path):
    _corpus(tmp_path)
    _coder_case(tmp_path, "canary-unstamped", split=None, warmup_arithmetic="composed")
    root = cases_root(tmp_path / "workspace")
    before = _snapshot(root)

    result = runner.invoke(app, ["bench", "corpus", "--site", "coder"])

    assert result.exit_code == 0, result.output
    assert _snapshot(root) == before


# ── the corpus this repo actually ships ──────────────────────────────────────────────────────


def test_the_committed_coder_corpus_reports_its_two_shipped_buckets_and_their_balance():
    """The real files in ``cases/coder/``, through the real loader — no fixture, no copy."""
    cases = CoderCaseProvider(cases_root=COMMITTED_CASES).load(CODER_SITE_ID)

    rendered = render_corpus_report(
        read_corpus(CODER_SITE_ID, cases, vocabulary=_committed_vocabulary()),
        source=COMMITTED_CASES / CODER_SITE_ID,
    )

    buckets = _block(rendered, "Buckets")
    assert _row(buckets, "canary").startswith("6 case(s)")
    assert _row(buckets, "edge").startswith("14 case(s)")
    assert _row(_block(rendered, "Split balance"), "corpus").endswith("unstamped 0")


def test_every_axis_of_the_committed_coder_corpus_accounts_for_every_case_it_ships():
    """A stratification that lost a case would be a stratification nobody could reason with."""
    cases = CoderCaseProvider(cases_root=COMMITTED_CASES).load(CODER_SITE_ID)

    reading = read_corpus(CODER_SITE_ID, cases, vocabulary=_committed_vocabulary())

    assert reading.case_count == len(cases)
    for axis in reading.axes:
        counted = sum(level.balance.cases for level in axis.levels)
        assert counted == reading.case_count, axis.axis


def _committed_vocabulary() -> CorpusVocabulary:
    """The coder site's own labelling vocabulary, through the eval layer's site lookup."""
    return site_vocabulary(CODER_SITE_ID)
