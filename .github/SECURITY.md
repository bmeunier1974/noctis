# Security Policy

Noctis is **paper-only research software**. Its defining safety property — that no real-money
order path is reachable — is enforced structurally, and we treat that boundary as a security
surface (see [Scope](#scope) below).

## Supported Versions

Security fixes are applied to the latest released minor series. Noctis is pre-1.0, so only the
most recent release line receives fixes.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a Vulnerability

**Please do not report security issues in public GitHub issues, pull requests, or
discussions.** Public disclosure before a fix puts users at risk.

Report privately through **GitHub Private Vulnerability Reporting**:

> **Security** tab → **Report a vulnerability**

This opens a private advisory visible only to the maintainers — no email address required, and
your report stays confidential until a fix is published. (If Private Vulnerability Reporting is
not enabled on a given fork or mirror, contact a maintainer privately through their GitHub
profile rather than filing a public issue.)

### What to include

- The affected version or commit.
- A clear description of the issue and its impact.
- Minimal steps to reproduce (a failing test, a config, or a short script is ideal).
- Any suggested remediation, if you have one.

### What to expect

- We aim to acknowledge a report within a few days and to keep you updated as we investigate.
- Once a fix is ready we will publish it, note the advisory, and credit you if you wish.

## Scope

Because Noctis is paper-only *by construction*, the paper-only guarantee is itself a security
boundary. In particular, **treat anything that weakens the two-gate invariant as a reportable
vulnerability**:

- Any path that could reach a real-money order without **both** `mode: live` (config) **and**
  `ALLOW_LIVE=true` (environment) set — the two gates deliberately live in separate sources
  (`src/noctis/config/gate.py`).
- Any change that lets `mode: live` silently downgrade to paper instead of failing at startup.
- Any code that wires a real order path into the live adapter stub (which is designed to refuse
  execution).
- Lookahead or holdout leakage that lets future/holdout data reach a trading or promotion
  decision — a correctness-and-integrity boundary of the research pipeline.

Also in scope: leaked secrets, dependency vulnerabilities with a practical exploit path, and
anything that would cause credentials or vendor market data to be committed to the repository.

Out of scope: the inherent risk of trading strategies losing money in simulation — Noctis makes
no claim of profitability (see the [disclaimer](README.md#disclaimer)).
