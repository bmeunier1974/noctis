---
summary: Tune and decide the EXISTING strategy library first; author new files only after a completed tune-to-verdict cycle.
config:
  promotion:
    metric: sortino
---
<!--
  A session-conduct mandate for backends that cannot reliably clear the write gate
  (small local models fixate on authoring files that fail validation and never reach a
  backtest — see MEMORY.md). It lives at the top level of mandate/, NOT in profiles/,
  on purpose: profiles/ is the `auto` mode's menu of trader personalities, and this is
  a conduct prior, not a personality. Select it with `research.mandate: tune-first`.

  The metric overlay mirrors MANDATE.md (sortino) deliberately: the champion board was
  elected under sortino, and a session scored on a different metric would mark every
  sitting champion stale (displaceable). Keep the two in lockstep if you change one.
-->

Spend this session tuning and DECIDING what the strategy library already holds, not
authoring new code. Start from the existing strategies (list_strategies): pick the most
promising non-rejected one — judged against the market digest, memory, and the champion
board — baseline it with run_backtest on its current params, explore with run_sweep, and
drive it to an explicit verdict (evaluate_vs_champion or reject_strategy). One completed
tune-to-verdict cycle on an existing strategy is worth more than any number of unfinished
drafts.

Author a new file only after you have completed at least one tune-to-verdict cycle this
session, and even then prefer the smallest possible step: revise an existing PASSING file,
or adapt the shipped template with minimal changes, rather than writing from scratch. If a
write_strategy submission is rejected twice, stop authoring and return to tuning — the
write gate is telling you the code does not match its own declared thesis, and budget spent
re-submitting is budget not spent producing evidence.

Stay with each strategy's own researched symbols and declared style unless the experiment
log gives a concrete reason to widen; symbol discovery and brand-new theses belong to
sessions running under a personality mandate, not this one.
