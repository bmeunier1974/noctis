"""Noctis — an autonomous, paper-only trading agent.

Researches strategies while the market is closed, emits paper orders while it is open,
reports at close, and keeps
its own memory. Real-money execution is gated behind two independent switches and is
unreachable in the default paper mode.
"""

__version__ = "0.1.0"
