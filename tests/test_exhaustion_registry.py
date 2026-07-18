"""The cross-session exhausted-class registry — record, match, persist, summarize."""

from __future__ import annotations

import pytest

from noctis.research.exhaustion_registry import ExhaustedClassRegistry


def test_record_and_match_is_case_and_whitespace_insensitive(tmp_path):
    reg = ExhaustedClassRegistry(tmp_path / "ex.json")
    reg.record("Per-Symbol   Long/Flat MA Overlay", "forfeits drift", example="ma_cross")
    assert reg.is_exhausted("per-symbol long/flat ma overlay") is not None
    assert reg.is_exhausted("PER-SYMBOL LONG/FLAT MA OVERLAY") is not None
    assert reg.is_exhausted("a different class") is None


def test_record_upserts_examples_and_refreshes_reason(tmp_path):
    reg = ExhaustedClassRegistry(tmp_path / "ex.json")
    reg.record("class x", "first reason", example="a")
    reg.record("class x", "second reason", example="b")
    reg.record("class x", "second reason", example="a")  # duplicate example ignored
    rec = reg.is_exhausted("class x")
    assert rec["reason"] == "second reason"
    assert rec["examples"] == ["a", "b"]
    assert len(reg.load()) == 1  # still a single class record


def test_persists_across_instances(tmp_path):
    path = tmp_path / "ex.json"
    ExhaustedClassRegistry(path).record("dead class", "why", example="s1")
    # A fresh instance (a new session/process) reads the same file.
    assert ExhaustedClassRegistry(path).is_exhausted("dead class") is not None


def test_missing_or_corrupt_file_reads_as_empty(tmp_path):
    assert ExhaustedClassRegistry(tmp_path / "nope.json").load() == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert ExhaustedClassRegistry(bad).load() == []
    assert ExhaustedClassRegistry(bad).is_exhausted("x") is None


def test_summary_truncates_reason_and_keeps_label_and_examples(tmp_path):
    reg = ExhaustedClassRegistry(tmp_path / "ex.json")
    reg.record("Fancy Class", "R" * 500, example="s")
    (row,) = reg.summary(reason_chars=50)
    assert row["class_tag"] == "Fancy Class"  # original casing preserved for the human/agent
    assert len(row["reason"]) <= 50 and row["reason"].endswith("…")
    assert row["examples"] == ["s"]


def test_empty_tag_is_rejected(tmp_path):
    reg = ExhaustedClassRegistry(tmp_path / "ex.json")
    with pytest.raises(ValueError):
        reg.record("   ", "reason")
    assert reg.is_exhausted("   ") is None
