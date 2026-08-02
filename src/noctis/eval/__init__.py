"""The eval layer — benchmark infrastructure for the engine's LLM judgment sites.

**The boundary, stated where it would be broken: this package is one-way.** The eval layer imports
the engine; the engine never imports the eval layer. If you are here because a module under
``src/noctis`` wants something that lives in ``noctis.eval`` — a site declaration, a harness spec,
a scorer — the answer is not an import. Either the thing belongs in the engine (move it there and
have the eval layer import *it*), or the engine does not need it. There is no third option, and the
guard below will not let you discover this later.

**Why it is a rule and not a preference.** A benchmark exists to measure production, so production
must not be able to notice that a benchmark exists. The moment an engine module imports this
package, a bench-only ablation ("run FORMULATE with the contract sheet off, to see what it is
worth") becomes reachable from a real research session, and every run afterwards is a run whose
prompt composition nobody can state from the record alone. That is the platform's design invariant
3 — production behaviour never depends on benchmark infrastructure — and like every load-bearing
rule in this repo it is enforced structurally rather than by convention.

**The enforcement.** :mod:`noctis.eval.guard` is the import-isolation guard: a static, stdlib-only
scan of a source tree that returns every module *outside* this package which imports it. The suite
runs it against ``src/noctis`` on every CI run and fails hard, naming the offending modules — the
same posture as the engine-fingerprint ratchet and the config overlay's gate-subtree assertion,
which raise rather than warn because a guard that warns is a guard that gets ignored.

The package is deliberately thin. Beside the boundary and its guard — here from day one, because
the invariant is cheapest to hold from the first commit — it carries only declarations:
:mod:`noctis.eval.site` (``AgentSite``, one frozen record per LLM judgment site),
:mod:`noctis.eval.knobs` (``SiteKnobs``, the typed knob set an override is checked against),
:mod:`noctis.eval.harness` (``HarnessSpec``, the eval-only prompt-composition overlay that
production config has no word for) and :mod:`noctis.eval.registry` (the plain id→site lookup the
declarations populate as code). The five site declarations and the runner that invokes them arrive
in later stories; nothing here runs anything yet.
"""
