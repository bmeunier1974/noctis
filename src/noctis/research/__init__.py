"""Agent-first research.

Two seams live here:

* :mod:`noctis.research.agent` + :mod:`noctis.research.tools` — **the agent loop**: Claude
  authors one-file Python strategies into the library and drives formulate → match →
  optimize → decide through a curated tool registry (journal + exhaustion gate included).
* :mod:`noctis.research.ideation` — the legacy LLM ideation of ``StrategySpec`` families
  that feed the proposer/Optuna loop.

Both ride the provider-neutral :mod:`noctis.research.llm` seam and degrade gracefully to
no-ops without the ``[llm]`` extra or a key for the configured provider.
"""

from __future__ import annotations

from .agent import run_agent_research
from .author import AuthoringError, StrategyAuthor, StrategyBrief
from .cost import CostProfile, resolve_budgets, resolve_cost_profile
from .ideation import IdeationContext, Ideator, build_ideator, propose_specs
from .llm import (
    Capabilities,
    ClientStatus,
    LiteLLMClient,
    LLMClient,
    ToolCall,
    Turn,
    build_llm_client,
    client_for,
    client_status,
    effective_web_search,
    provider_of,
    thinking_for,
)
from .mandate import (
    Mandate,
    MandateError,
    Reference,
    apply_overrides,
    profiles_catalog,
    resolve_mandate,
)
from .prompt import build_system_prompt
from .sweep import SweepRunner
from .tools import ResearchToolbox

__all__ = [
    "Ideator",
    "IdeationContext",
    "build_ideator",
    "propose_specs",
    "ResearchToolbox",
    "StrategyAuthor",
    "StrategyBrief",
    "AuthoringError",
    "SweepRunner",
    "run_agent_research",
    "build_system_prompt",
    "build_llm_client",
    "client_for",
    "client_status",
    "ClientStatus",
    "LLMClient",
    "LiteLLMClient",
    "Capabilities",
    "Turn",
    "ToolCall",
    "provider_of",
    "thinking_for",
    "effective_web_search",
    "CostProfile",
    "resolve_cost_profile",
    "resolve_budgets",
    "Mandate",
    "MandateError",
    "Reference",
    "resolve_mandate",
    "apply_overrides",
    "profiles_catalog",
]
