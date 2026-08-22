"""The two sites whose output no emit contract types: the **coder** and the memory **distiller**.

Every other declared site answers with a typed record the episodic driver parses through an
:class:`~noctis.research.episode.EmitContract`. These two do not, and the declaration model says so
rather than inventing a contract that production does not have:

* the **coder** replies with one fenced code block, and the arbiter of whether it was a good answer
  is the fresh-subprocess write gate (:func:`noctis.strategies.library.write_strategy`) — a file
  that imports, replays its scenarios and survives the Tier-1 structural invariants. There is no
  schema between the model and the verdict, so :attr:`~noctis.eval.site.AgentSite.contract` is
  ``None`` and the gate stays the only judge;
* the **distiller** replies with bullet lines, of which production keeps the ones starting ``- ``,
  capped at the caller's line budget. Also no schema, also ``None``.

**Both renderers are adapters, and an adapter forwards.** Neither reimplements a prompt: the coder
adapter asks the live :class:`~noctis.research.author.StrategyAuthor` for the two halves it is about
to send, and the distill adapter formats :mod:`noctis.research.distill`'s own template object. What
is rendered here is therefore what a real session sends, byte for byte — which is the property
``tests/test_eval_coder_distill_sites.py`` pins by driving the production engines and comparing.

**The :class:`~noctis.eval.harness.HarnessSpec` is accepted and not yet read** by either adapter.
Wiring the ablation dials (contract sheet off, template off, worked example swapped, feasibility
rules off, retry hints off) into the prompt-composition path is the eval-runner epic's job; until
then a spec is carried for signature compatibility only, and every ablation renders exactly as
production does. Better an honestly inert parameter than a second prompt assembly living here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from noctis.eval.coder_scorer import CODER_SCORER
from noctis.eval.harness import HarnessSpec
from noctis.eval.knobs import SiteKnobs
from noctis.eval.site import AgentSite
from noctis.research import distill
from noctis.research.author import StrategyAuthor, StrategyBrief
from noctis.strategies.library import StrategyValidationError
from noctis.strategies.scenario_spec import SpecSuite

__all__ = [
    "CODER_SITE",
    "DISTILL_SITE",
    "NO_PROMPT",
    "AuthoringJob",
    "CoderKnobs",
    "DistillKnobs",
    "FindingsHistory",
    "render_coder_prompt",
    "render_distill_prompt",
]

# The coder's contract generation. Hand-bumped when the site's *interface* changes shape — a new
# required field on :class:`~noctis.research.author.StrategyBrief`, or a write gate that judges
# something other than one complete strategy file — because that is when two benchmark records stop
# being comparable. Ordinary prompt wording moves under the prompt-asset ratchet's clock
# (``prompt_fingerprint.json`` + ``docs/prompt-changelog.md``), not this one: the wording is
# *measured* by a benchmark, so re-versioning on every edit would make every run its own generation.
CODER_VERSION = "1"

# The distiller's contract generation, on the same hand-bumped discipline: findings history in,
# bullet lessons out. Bump it when that shape changes, not when the wording does.
DISTILL_VERSION = "1"

# How the coder's two production halves are joined into the single string a renderer returns. The
# transport sends them as separate fields (system, then one user message), so the adapter's whole
# composition is this join, in that order — nothing is re-ordered, filtered or rewritten.
_SYSTEM_USER_JOIN = "\n\n"

#: What a rendered coder prompt says when the engine refused the brief before composing anything —
#: an unresolvable ``reference``. Marked as a comment so nothing downstream mistakes it for a
#: prompt, and followed by the engine's own refusal, which is the whole diagnosis.
NO_PROMPT = "# no prompt was composed — the engine refused this brief before asking:"


@dataclass(frozen=True)
class AuthoringJob:
    """One authoring job, as the coder site is asked it: an engine, a name, and the brief.

    The engine is part of the *input* because it carries the install-derived half of the prompt —
    the contract sheet, the shipped ``TEMPLATE.py`` and the worked-example seed read off disk at
    construction, plus its client's capabilities. A benchmark that wants to vary that half varies
    the engine it passes; a benchmark that wants to vary the research judgment varies the brief.

    ``spec`` is the compiled fixed scenario oracle (``None`` ⇒ the coder owns tape construction and
    reads the brief's prose sketch), and ``prior`` is a previous ``(source, validation error)``
    pair — the private-retry material, so the retry prompt is renderable as its own site input
    rather than only reachable from inside a failed run.
    """

    author: StrategyAuthor
    name: str
    brief: StrategyBrief
    spec: SpecSuite | None = None
    prior: tuple[str, str] | None = None


@dataclass(frozen=True)
class FindingsHistory:
    """What the distiller is shown: the full findings history, oldest first, and its line budget.

    Production folds *the whole history* — never a tail — and caps the answer at ``max_lines``,
    which is an argument of the distillation call rather than a setting, so it rides on the input.
    """

    findings: tuple[str, ...]
    max_lines: int = distill._DISTILL_MAX_LINES


@dataclass(frozen=True)
class CoderKnobs(SiteKnobs):
    """The coder-model settings family, exactly as ``research.agent`` declares it today.

    Ten fields, ten existing knobs — model selection, the paid fallback and its escalation cap, the
    two thinking dials, the two sampling levers, the output-token ceiling, the private
    validator-retry budget and the per-session author-call budget. The defaults are the shipped
    ones, so an un-overridden knob set *is* production.

    Nothing here is invented: ``coder_temperature``, ``coder_seed`` and ``coder_retries`` are knobs
    production grew (#222) — the sampling pair rides the shared client builder, the retry budget
    pins what was a module constant — and a declaration follows the settings model rather than
    leading it. A lever production still has no word for (a per-attempt timeout, say) stays a
    parameter of whoever drives the engine and is *not* promoted here: the knob set is what a
    benchmark override is validated against, and an override may only name a real one.
    """

    coder_model: str | None = None
    coder_fallback_model: str | None = None
    max_escalations: int = 0
    coder_thinking: Literal["off", "on"] = "on"
    coder_fallback_thinking: Literal["off", "on"] = "off"
    coder_max_tokens: int | None = None
    coder_temperature: float | None = None
    coder_seed: int | None = None
    coder_retries: int | None = None
    max_author_calls: int | None = None


@dataclass(frozen=True)
class DistillKnobs(SiteKnobs):
    """Distillation's whole config surface: one cadence knob, and honestly nothing else.

    Everything else that shapes a distillation is a module constant in
    :mod:`noctis.research.distill` — the line cap, the token ceiling, the minimum history worth
    folding — and a declaration promotes none of them to config to look better furnished. Even the
    model is not a knob of this site's own: the close-phase trigger reuses the coder's paid fallback
    when one is configured, so that lever is declared where it lives, on the coder site.
    """

    memory_distill_every: int = 0


def render_coder_prompt(job: AuthoringJob, spec: HarnessSpec) -> str:
    """The exact prompt the coder is sent for ``job``: the engine's two halves, joined.

    A forwarding adapter and nothing more. The system half is the byte-stable prefix the engine
    built at construction (role rules, contract sheet, feasibility rules, ``TEMPLATE.py``, the
    worked example, output rules) — the same string ``cached_system`` wraps for transport without
    changing a byte — and the user half is the engine's own per-attempt prompt, asked for with the
    reference/revision material resolved the way an authoring call resolves it. The two are joined
    in transport order; nothing is re-ordered, filtered or rewritten.

    **A brief production refuses composes nothing, and this says so** (:data:`NO_PROMPT`). The
    engine rejects a ``reference`` the library does not ship *before* it spends a completion — a bad
    brief, not a coder failure — so there are no bytes to reproduce, and the corpus deliberately
    ships such a case (its honest outcome is a refusal rather than a file). Returning the engine's
    own refusal keeps this renderer total for every case a corpus can hold: inventing a prompt
    production would never send would be the only worse answer, and raising would turn one case's
    honest failure into a bench that never finished the other nineteen.

    ``spec`` is accepted and not read: the ablation dials are wired in the eval-runner epic (see
    the module docstring), so every :class:`~noctis.eval.harness.HarnessSpec` renders production's
    composition today.
    """
    del spec  # honestly inert until the ablation wiring lands
    author = job.author
    try:
        reference_source = author._reference_source(job.brief)
    except StrategyValidationError as refusal:
        return f"{NO_PROMPT}\n{refusal}\n"
    user = author._user_prompt(
        job.name,
        job.brief,
        job.prior,
        reference_source=reference_source,
        current_source=author._current_source(job.name),
        spec=job.spec,
    )
    return f"{author._system_prompt}{_SYSTEM_USER_JOIN}{user}"


def render_distill_prompt(history: FindingsHistory, spec: HarnessSpec) -> str:
    """The exact prompt one distillation folds ``history`` with.

    Distillation's assembly is a single ``format`` of one module-level template inline in
    :func:`~noctis.research.distill.distill_findings`, so there is no function to forward to: the
    adapter binds that template object by identity and makes the same two substitutions production
    makes, in the same order. Should the assembly ever be extracted into a function, bind the
    function here instead — the byte-identity test holds either way, because it compares against
    what the production call actually sent.

    ``spec`` is accepted and not read, exactly as in :func:`render_coder_prompt`.
    """
    del spec  # honestly inert until the ablation wiring lands
    return distill._PROMPT.format(max_lines=history.max_lines, findings="\n".join(history.findings))


# The declarations themselves — data, assembled at import, running nothing.
CODER_SITE: AgentSite[AuthoringJob, str] = AgentSite(
    id="coder",
    version=CODER_VERSION,
    # No emit contract, on purpose: the fresh-subprocess write gate is this site's judge.
    contract=None,
    render=render_coder_prompt,
    knobs=CoderKnobs,
    # The site's own reading over the retained job records: two co-primary pass rates, the effort
    # and spend behind them, the failure taxonomy and the detectors' advisory rows (#226).
    scorers=(CODER_SCORER,),
)

DISTILL_SITE: AgentSite[FindingsHistory, tuple[str, ...]] = AgentSite(
    id="distill",
    version=DISTILL_VERSION,
    # No emit contract either: the answer is bullet lines, filtered and capped by the caller.
    contract=None,
    render=render_distill_prompt,
    knobs=DistillKnobs,
)
