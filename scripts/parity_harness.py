#!/usr/bin/env python
"""Parity harness — conversation vs episodic on one fixed lake fixture (epic #62, story #75).

The evidence gate for flipping ``auto`` to episodic (#76). This is a **dev tool, not a CLI
subcommand**: the operator supplies a hosted API key and a synced lake fixture, runs the script,
and reads a side-by-side comparison of the two research loops on the SAME model, fixture, and
mandate. It runs paid model sessions, so it refuses without a key and prints what it is about to
spend before it spends it.

    uv run python scripts/parity_harness.py --help
    uv run python scripts/parity_harness.py --sessions 3 --model anthropic/claude-3-5-haiku

The metric computation and the flip-criterion verdict live in :mod:`noctis.research.parity` (pure,
tested). This file is a thin orchestrator: resolve settings once through the composition root
(:func:`noctis.bootstrap.resolve_session`), then for each loop force ``research.agent.loop``,
assemble a session with the same bootstrap builders ``noctis research`` uses, run N sessions,
collect ``(summary, rollup)`` pairs, and hand them to the parity module to compute and render. It
parses no transcript — every number comes from the summary and the ledger rollup.

See ``docs/parity.md`` for prerequisites, how to read the table, and the flip criterion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noctis.research.parity import SessionPair

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    # Let the script run from a source checkout without an editable install.
    sys.path.insert(0, str(_SRC))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="parity_harness",
        description=(
            "Run both research loops (conversation, episodic) on the same model, lake fixture, and "
            "mandate, and print a side-by-side metrics comparison for the auto-flip evidence gate "
            "(#76). Runs PAID model sessions — needs a hosted API key and a synced lake fixture."
        ),
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="Sessions to run PER loop (total paid sessions = 2 x this). Default 1.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override research.model for both loops (e.g. anthropic/claude-3-5-haiku). The whole "
        "point is one hosted model on both sides; defaults to the configured research.model.",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to a config YAML.")
    parser.add_argument(
        "--mandate",
        default=None,
        help="Select a mandate under mandate_dir for every session (mutually exclusive "
        "with --directive).",
    )
    parser.add_argument(
        "--directive",
        "-d",
        default=None,
        help="Inline operator mandate for every session (mutually exclusive with --mandate).",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Scoring metric for every session: sharpe | sortino | total_return.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Tool rounds (conversation) / episodes (episodic) per session; default from config.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Stream each session's tool feed (-v) / reasoning (-vv) while it runs.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the spend confirmation prompt (for a non-interactive run).",
    )
    return parser.parse_args(argv)


def _run_loop(loop: str, args: argparse.Namespace) -> Sequence[SessionPair]:
    """Run ``args.sessions`` sessions of one loop and collect ``(summary, rollup)`` pairs.

    A fresh session is assembled per run through the same bootstrap builders ``noctis research``
    uses, with ``research.agent.loop`` forced to ``loop`` so both loops share the settings, model,
    fixture, and mandate. The episodic rollup is loaded back from each summary's ledger path; the
    conversation loop writes none, so its rollup is ``None``."""
    from noctis.bootstrap import (
        build_console,
        build_families,
        build_lake,
        build_memory,
        build_research_session,
        resolve_session,
    )
    from noctis.champions import build_registry
    from noctis.research.parity import rollup_for

    pairs: list[SessionPair] = []
    for i in range(args.sessions):
        inputs = resolve_session(
            args.config,
            directive=args.directive,
            mandate=args.mandate,
            metric=args.metric,
        )
        settings = inputs.settings
        if args.model:
            settings.research.model = args.model
        settings.research.agent.loop = loop
        session = build_research_session(
            settings=settings,
            lake=build_lake(settings),
            registry=build_registry(settings),
            families=build_families(settings),
            memory=build_memory(settings),
            mandate=inputs.mandate,
            on_event=build_console(args.verbose),
        )
        if session is None:  # defensive: the key check already gated this
            raise SystemExit("no LLM client buildable for the configured research model")
        print(f"  [{loop}] session {i + 1}/{args.sessions} (model={session.model})...")
        summary = session.run(max_iterations=args.max_iterations)
        print(
            f"    -> {summary.stopped_reason}: {summary.promotions} promoted, "
            f"{summary.rejections} rejected, {summary.tokens_total} tokens"
        )
        pairs.append((summary, rollup_for(summary)))
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from noctis.bootstrap import resolve_session
    from noctis.research import client_status
    from noctis.research.parity import (
        CONVERSATION,
        EPISODIC,
        compute_loop_metrics,
        render_comparison,
    )

    if args.sessions < 1:
        print("--sessions must be >= 1", file=sys.stderr)
        return 2

    # Resolve settings once just to check the key and echo the plan; each session re-resolves.
    inputs = resolve_session(
        args.config, directive=args.directive, mandate=args.mandate, metric=args.metric
    )
    settings = inputs.settings
    if args.model:
        settings.research.model = args.model

    status = client_status(settings)
    if not status.ok:
        print(
            f"Refusing to run: no LLM client for model {status.model!r} ({status.reason}).\n"
            "This harness runs PAID model sessions — configure the [llm] extra and the provider's "
            "API key first. See docs/parity.md.",
            file=sys.stderr,
        )
        return 1

    total = args.sessions * 2
    mandate_source = inputs.mandate.source if inputs.mandate else "-"
    print(
        f"About to run {total} PAID model session(s): {args.sessions} per loop "
        f"(conversation + episodic) on model={status.model}, "
        f"lake={settings.data.lake_dir}, mandate={mandate_source}."
    )
    if not args.yes and sys.stdin.isatty():
        if input("Proceed and spend? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Aborted — no spend.")
            return 0

    print("\nRunning conversation loop...")
    conversation = _run_loop(CONVERSATION, args)
    print("\nRunning episodic loop...")
    episodic = _run_loop(EPISODIC, args)

    print("\n" + "=" * 58)
    print(
        render_comparison(
            compute_loop_metrics(CONVERSATION, conversation),
            compute_loop_metrics(EPISODIC, episodic),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
