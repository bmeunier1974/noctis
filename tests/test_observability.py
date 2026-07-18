"""The observability seam (P1): typed :class:`Event`s + the level-aware :class:`Console`.

Pure, no I/O — a fake sink collects the console's output so level gating, colorization, the
width cap, and string back-compat are asserted directly.
"""

from __future__ import annotations

import pytest

from noctis.observability import Console, Event, render_plain


def _console(verbose, **kw):
    """A Console whose output lands in a list instead of the terminal."""
    out: list[str] = []
    kw.setdefault("color", False)
    return Console(verbose, sink=out.append, **kw), out


# ── level gating ─────────────────────────────────────────────────────────────────────────────
def test_level_gates_think_and_say_to_level_two():
    con, out = _console(1)
    con(Event("tool", "run_backtest(...) -> ok", level=1))
    con(Event("think", "reasoning here", level=2))
    con(Event("say", "narration here", level=2))
    # At -v: the level-1 tool line shows; the level-2 reasoning/narration do not.
    assert any("run_backtest" in line for line in out)
    assert not any("reasoning here" in line for line in out)
    assert not any("narration here" in line for line in out)


def test_level_two_opens_the_firehose():
    con, out = _console(2)
    con(Event("think", "reasoning here", level=2))
    con(Event("usage", "tokens in=1 out=2", level=2))
    assert any("reasoning here" in line for line in out)
    assert any("tokens in=1" in line for line in out)


def test_show_reasoning_forces_think_and_say_without_full_verbosity():
    # -v (level 1) plus --show-reasoning: think/say show, but other level-2 noise (usage) does not.
    con, out = _console(1, show_reasoning=True)
    con(Event("think", "reasoning here", level=2))
    con(Event("say", "narration here", level=2))
    con(Event("usage", "tokens in=1 out=2", level=2))
    assert any("reasoning here" in line for line in out)
    assert any("narration here" in line for line in out)
    assert not any("tokens in=1" in line for line in out)  # usage stays -vv-only


def test_saw_think_tracks_only_surfaced_think_events():
    # Gated out ⇒ not counted; surfaced ⇒ counted (drives the CLI degradation hint).
    con, _ = _console(1)  # think is level-2, so -v gates it out
    con(Event("think", "reasoning", level=2))
    assert con.saw_think is False
    con2, _ = _console(2)
    con2(Event("think", "reasoning", level=2))
    assert con2.saw_think is True


# ── color + width ────────────────────────────────────────────────────────────────────────────
def test_color_off_emits_no_escape_codes():
    con, out = _console(2, color=False)
    con(Event("think", "reasoning here", level=2))
    con(Event("tool", "run_backtest(...) -> ERROR: boom", meta={"ok": False}, level=1))
    assert out and all("\x1b[" not in line for line in out)


def test_color_on_wraps_error_results_in_a_red_escape():
    con, out = _console(1, color=True)
    con(Event("tool", "run_backtest(...) -> ERROR: boom", meta={"ok": False}, level=1))
    assert out[0].startswith("\x1b[")  # styled
    assert "\x1b[31m" in out[0] or "\x1b[91m" in out[0]  # red for an ok=False result


def test_no_color_env_disables_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    con, out = _console(1, color=True)  # asked for color, but NO_COLOR wins
    con(Event("tool", "x -> ok", level=1))
    assert "\x1b[" not in out[0]


def test_width_cap_wraps_long_lines_and_aligns_continuations():
    con, out = _console(2, width=40, color=False)
    long_text = "word " * 40  # far wider than 40 columns
    con(Event("think", long_text.strip(), level=2))
    lines = out[0].splitlines()
    assert len(lines) > 1  # wrapped
    assert all(len(line) <= 40 for line in lines)  # every physical line within the cap
    assert lines[0].startswith("🧠 ")  # first line gets the glyph gutter
    assert lines[1].startswith("  ") and not lines[1].startswith("🧠")  # continuations indent


# ── P5: in-place token streaming ───────────────────────────────────────────────────────────────
def _stream_console(verbose=2, *, tty=True, **kw):
    """A Console whose block output and raw stream output land in separate lists."""
    blocks: list[str] = []
    stream: list[str] = []
    kw.setdefault("color", False)
    con = Console(verbose, sink=blocks.append, stream_sink=stream.append, tty=tty, **kw)
    return con, blocks, stream


def test_delta_streams_in_place_on_tty_and_drops_the_duplicate_block():
    con, blocks, stream = _stream_console()
    con.delta("think", "hel")
    con.delta("think", "lo world")
    con(Event("think", "hello world", level=2))  # the completed block for the same reasoning
    joined = "".join(stream)
    assert joined.startswith("🧠 ")  # glyph gutter written once, up front
    assert "hello world" in joined  # every delta rendered live
    assert joined.endswith("\n")  # the stream is finalized when its block arrives
    assert blocks == []  # the block is a duplicate of the live stream → dropped
    assert con.saw_think is True  # a streamed think still counts for the CLI's degradation hint


def test_delta_is_a_noop_off_tty_and_the_block_renders_normally():
    con, blocks, stream = _stream_console(tty=False)
    con.delta("think", "hi")  # off a TTY: streaming falls back to block mode
    con(Event("think", "hi", level=2))
    assert stream == []  # nothing streamed
    assert blocks and "hi" in blocks[0]  # the completed block rendered as usual
    assert con.saw_think is True


def test_delta_kind_switch_finalizes_the_previous_line():
    con, _blocks, stream = _stream_console()
    con.delta("think", "reasoning")
    con.delta("say", "narration")  # think → say closes the think line first
    joined = "".join(stream)
    assert "🧠 reasoning\n" in joined  # think line finalized before say began
    assert joined.endswith("narration")  # the say stream is still open (no block yet)


def test_streamed_block_is_dropped_but_an_unstreamed_one_still_shows():
    con, blocks, stream = _stream_console()
    con.delta("say", "streamed narration")
    con(Event("say", "streamed narration", level=2))  # dropped: already shown live
    con(Event("say", "fresh narration", level=2))  # never streamed → shows as a block
    assert "streamed narration" in "".join(stream)
    assert blocks == ["fresh narration"]


def test_delta_width_wraps_and_aligns_continuations():
    con, _blocks, stream = _stream_console(width=20)
    con.delta("think", "alpha beta gamma delta epsilon zeta eta theta")
    con(Event("think", "…", level=2))  # finalize the stream
    lines = "".join(stream).split("\n")
    assert len([ln for ln in lines if ln.strip()]) > 1  # wrapped past the width cap
    assert any(ln.startswith("  ") for ln in lines[1:])  # continuations indent under the gutter


def test_legacy_string_finalizes_an_open_stream_first():
    con, blocks, stream = _stream_console()
    con.delta("think", "mid-thought")
    con("[misfire] tool call written as text markup")  # a legacy line interrupts the stream
    assert "".join(stream).endswith("\n")  # the open stream was closed before the line printed
    assert blocks == ["[misfire] tool call written as text markup"]


# ── string back-compat ───────────────────────────────────────────────────────────────────────
def test_bare_string_is_passed_through_verbatim():
    """A legacy emit site (misfire note, web_search auto-disable) still hands the sink a plain
    pre-formatted string, unchanged and always shown."""
    con, out = _console(1, color=True)
    con("[misfire] tool call written as text markup — asking for a native re-issue")
    assert out == ["[misfire] tool call written as text markup — asking for a native re-issue"]


# ── plain renderer (log fallback) ────────────────────────────────────────────────────────────
def test_render_plain_collapses_multiline_and_prefixes_by_kind():
    line = render_plain(Event("think", "first line\n  second line", level=2))
    assert line == "🧠 first line second line"  # glyph + single collapsed line
    assert render_plain(Event("say", "just narration", level=1)) == "just narration"  # no glyph


@pytest.mark.parametrize("kind", ["think", "say", "tool", "result", "usage", "trade", "refuse"])
def test_render_plain_is_single_line_for_every_kind(kind):
    assert "\n" not in render_plain(Event(kind, "a\nb\nc"))


# ── P6: in-place activity heartbeat ────────────────────────────────────────────────────────────
def test_activity_frame_formats_spinner_label_and_clock():
    con, _ = _console(1, width=100)
    assert con._activity_frame("optimize foo", 47, 0) == "⠋ optimize foo · 0:47"
    assert con._activity_frame("x", 0, 1).startswith("⠙ ")  # the frame glyph cycles by index
    assert "1:05" in con._activity_frame("x", 65, 0)  # elapsed rolls over to minutes:seconds


def test_activity_frame_truncates_to_width():
    con, _ = _console(1, width=20)
    frame = con._activity_frame("a-very-long-label-that-overflows", 3, 0)
    assert len(frame) <= 20 and frame.endswith("…")


def test_activity_off_tty_prints_one_breadcrumb_and_no_spinner():
    # Off a TTY (a pipe/log) the spinner can't animate, so a single label breadcrumb stands in.
    con, out = _console(1, tty=False)
    with con.activity("optimize donchian_breakout"):
        pass
    assert out == ["⠿ optimize donchian_breakout …"]


def test_activity_on_tty_animates_in_place_then_erases(monkeypatch):
    import time as _time

    from noctis.observability import console as console_mod

    monkeypatch.setattr(console_mod, "_ACTIVITY_INTERVAL", 0.01)  # paint fast so the test is quick
    con, _blocks, stream = _stream_console(verbose=1, tty=True)
    with con.activity("optimize donchian_breakout"):
        _time.sleep(0.05)  # let the heartbeat thread paint a few frames
    joined = "".join(stream)
    assert "optimize donchian_breakout" in joined  # at least one live frame painted
    assert "\r" in joined  # frames repaint in place via carriage return
    assert stream[-1] == "\r\x1b[K"  # the spinner is erased on exit; the result Event prints next
