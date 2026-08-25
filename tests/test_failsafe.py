"""The fail-safe latch — one trip/warn-once/latch-off contract every sidecar writer composes.

External behaviour only (story #348, epic #341): what a guarded call returns, what lands through
the injected writer, what reaches the log, and that nothing an observability writer does can ever
raise into the engine that called it. The writer is the seam — every test here hands the latch a
callable that writes (or refuses to), so a failing disk is injected, never monkeypatched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from noctis.observability.failsafe import FailSafe, write_text

LATCH_LOGGER = "noctis.test.failsafe"

SUBJECT = "test writer alpha"
CONSEQUENCE = "nothing further will be written this session"


def _latch(**kwargs) -> FailSafe:
    """A latch under a test-owned logger, so `caplog` sees only this module's warnings."""
    return FailSafe(
        logger=logging.getLogger(LATCH_LOGGER),
        subject=SUBJECT,
        consequence=CONSEQUENCE,
        **kwargs,
    )


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LATCH_LOGGER and r.levelno == logging.WARNING]


def _boom(path: Path, payload: str) -> None:
    """A writer that refuses every write — the injected failing disk."""
    raise OSError("simulated disk failure")


# ── the happy path: a guarded call is transparent ─────────────────────────────────────────────


def test_a_guarded_call_returns_its_value_and_leaves_the_latch_closed(caplog):
    latch = _latch()

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        result = latch.guard(lambda n: n * 2, 21)

    assert result == 42  # the guard is transparent while nothing has failed
    assert latch.tripped is False
    assert _warnings(caplog) == []


def test_a_guarded_write_lands_the_payload_through_the_default_writer(tmp_path):
    latch = _latch()
    target = tmp_path / "sidecar.txt"

    latch.guard(latch.write, target, "a body — with an em dash")

    assert target.read_text(encoding="utf-8") == "a body — with an em dash"  # utf-8, once, here
    assert latch.tripped is False


def test_the_injected_writer_is_the_one_every_guarded_write_reaches(tmp_path):
    seen: list[tuple[Path, str]] = []
    latch = _latch(writer=lambda path, payload: seen.append((path, payload)))

    latch.guard(latch.write, tmp_path / "a.txt", "body")

    assert seen == [(tmp_path / "a.txt", "body")]
    assert list(tmp_path.iterdir()) == []  # the seam stands in for the disk entirely


# ── the trip: exactly one warning, naming what stopped ────────────────────────────────────────


def test_the_first_failure_warns_once_naming_what_stopped_and_latches_off(caplog):
    latch = _latch(writer=_boom, cause="a write failure")

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        result = latch.guard(latch.write, Path("nowhere.txt"), "body")

    assert result is None  # a latched call is told nothing was done
    assert latch.tripped is True
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert SUBJECT in message  # what stopped
    assert "self-disabled after a write failure" in message  # why
    assert "OSError: simulated disk failure" in message  # the failure itself, named
    assert CONSEQUENCE in message  # and what it means for the rest of the session


def test_the_cause_defaults_to_an_internal_failure(caplog):
    latch = _latch(writer=_boom)

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        latch.guard(latch.write, Path("nowhere.txt"), "body")

    assert "self-disabled after an internal failure" in _warnings(caplog)[0].getMessage()


def test_every_call_after_the_trip_is_a_silent_no_op(caplog):
    calls = {"n": 0}

    def work() -> str:
        calls["n"] += 1
        raise OSError("simulated disk failure")

    latch = _latch()
    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        first = latch.guard(work)
        later = [latch.guard(work) for _ in range(3)]

    assert first is None and later == [None, None, None]
    assert calls["n"] == 1  # the body was never entered again — a latch, not a retry
    assert len(_warnings(caplog)) == 1  # exactly one warning — no spam
    assert latch.tripped is True


def test_the_latch_never_clears_after_the_writer_recovers(tmp_path, caplog):
    fail = {"on": True}

    def flaky(path: Path, payload: str) -> None:
        if fail["on"]:
            raise OSError("simulated disk failure")
        write_text(path, payload)

    latch = _latch(writer=flaky)
    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        latch.guard(latch.write, tmp_path / "a.txt", "never lands")
        fail["on"] = False  # the disk recovers
        latch.guard(latch.write, tmp_path / "b.txt", "still never lands")

    assert list(tmp_path.iterdir()) == []  # one behaviour for the session, not half-coverage
    assert len(_warnings(caplog)) == 1


def test_a_failed_write_aborts_the_rest_of_its_guarded_body(tmp_path, caplog):
    # The write raises INTO its enclosing guard, so a body cannot carry on past a write that
    # never landed and leave a half-written artifact reading as a whole one.
    reached: list[str] = []
    latch = _latch(writer=_boom)

    def body() -> str:
        latch.write(tmp_path / "a.txt", "body")
        reached.append("after the write")
        return "done"

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        assert latch.guard(body) is None

    assert reached == []
    assert len(_warnings(caplog)) == 1


# ── nothing escapes (except an operator's own interrupt) ──────────────────────────────────────


@pytest.mark.parametrize(
    "exc", [OSError("disk"), ValueError("bad payload"), RuntimeError("worker wedged")]
)
def test_no_exception_escapes_a_guarded_body(exc, caplog):
    latch = _latch()

    def work() -> None:
        raise exc

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        assert latch.guard(work) is None  # whatever failed, the caller is never made to care

    assert latch.tripped is True
    assert type(exc).__name__ in _warnings(caplog)[0].getMessage()


def test_a_keyboard_interrupt_is_not_swallowed_by_the_latch(caplog):
    # Ctrl-C is the operator talking to the engine, not an internal failure: it rides straight
    # through, and it must not latch observability off on its way.
    latch = _latch()

    def work() -> None:
        raise KeyboardInterrupt

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        with pytest.raises(KeyboardInterrupt):
            latch.guard(work)

    assert latch.tripped is False
    assert _warnings(caplog) == []


# ── the trip note: a best-effort honesty stamp, still written after the latch is set ──────────


def test_the_trip_note_still_writes_through_the_latch_after_it_has_tripped(tmp_path, caplog):
    note = tmp_path / "note.txt"
    calls = {"n": 0}

    def writer(path: Path, payload: str) -> None:
        if path.name == "doomed.txt":
            raise OSError("simulated disk failure")
        write_text(path, payload)

    held: dict[str, FailSafe] = {}

    def trip_note() -> None:
        calls["n"] += 1
        held["latch"].write(note, "coverage stopped here")

    held["latch"] = latch = _latch(writer=writer, on_trip=trip_note)

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        latch.guard(latch.write, tmp_path / "doomed.txt", "body")
        latch.guard(latch.write, tmp_path / "later.txt", "body")  # a no-op: no second note

    assert note.read_text(encoding="utf-8") == "coverage stopped here"
    assert calls["n"] == 1  # the note is stamped once, when the latch trips
    assert len(_warnings(caplog)) == 1


def test_a_failing_trip_note_earns_no_second_warning(caplog):
    def note() -> None:
        raise OSError("the note could not be written either")

    latch = _latch(writer=_boom, on_trip=note)

    with caplog.at_level(logging.WARNING, logger=LATCH_LOGGER):
        latch.guard(latch.write, Path("nowhere.txt"), "body")

    assert latch.tripped is True  # best effort only: a failed note changes nothing
    assert len(_warnings(caplog)) == 1
