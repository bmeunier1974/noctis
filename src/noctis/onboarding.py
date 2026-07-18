"""First-run onboarding: the ``noctis setup`` wizard.

``noctis init`` copies templates and stops; ``setup`` goes the rest of the way to a machine
that can research and paper-trade *tonight*: scaffold the local files, install the optional
components, collect the DataBento key, connect an LLM (a hosted API key or a local
Ollama/noctis-ollama backend), prove the connection with one real completion, and say what
to run next. Every step is idempotent and edit-preserving — the wizard writes ``.env`` and
``config.yaml`` surgically (comments and unrelated lines survive), so re-running it is
always safe. ``--check`` runs the same probes read-only and exits non-zero on gaps, which
makes it the scriptable "is this install healthy?" doctor.

The interactive shell lives in :func:`run_setup`; everything it leans on (the dotenv/YAML
editors, the probes, the live LLM check) is a small module-level function so tests can
drive the wizard end-to-end with no network and no subprocesses.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import typer

# One representative module per optional extra: present ⇒ the extra is installed. These are
# the four seams `uv sync --all-extras` fills; find_spec never executes the module, so the
# probe stays instant even for the heavy stacks.
EXTRA_MODULES = {
    "llm": "litellm",  # the research/ideation LLM seam
    "data": "databento",  # lake ingests + the yfinance live feed
    "research": "vectorbt",  # backtest prefilter + optuna sweeps
    "engine": "nautilus_trader",  # the trading-node seam
}

OLLAMA_URL = "http://localhost:11434"
NOCTIS_OLLAMA_REPO = "https://github.com/bmeunier1974/noctis-ollama"

# The .env key per hosted-provider prefix. Everything else follows LiteLLM's own
# ``<PROVIDER>_API_KEY`` convention, so a key saved here is found by the seam.
_PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

_HOSTED_DEFAULT_MODEL = {"anthropic": "anthropic/claude-sonnet-5", "openai": "openai/gpt-5.4"}


# ─────────────────────────────────────────────────────────────────────────────
# Surgical file editors (pure enough to unit-test; never reformat, never drop lines)
# ─────────────────────────────────────────────────────────────────────────────
def set_env_key(path: Path, key: str, value: str) -> None:
    """Set ``KEY=value`` in a dotenv file, preserving every other line and comment.

    Replaces the first ``KEY=`` line (commented ``# KEY=`` lines are left alone), appends
    when the key is absent, creates the file when missing.
    """
    lines = path.read_text().splitlines() if path.is_file() else []
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def set_config_value(text: str, dotted_key: str, value: str) -> str:
    """Set ``section.sub.key: value`` in YAML text while preserving comments and layout.

    A deliberately minimal editor for the block-mapping shape the shipped template uses
    (two-space indents): the value on a matching key line is replaced in place with its
    inline comment kept; a missing key is appended at the end of its parent section;
    missing parent sections are appended at the end of the enclosing scope. ``value`` is
    inserted verbatim, so the caller formats it as a YAML scalar. This is what lets the
    wizard *write* config for the operator instead of asking them to merge a snippet into
    the file by hand.
    """
    lines = text.splitlines()
    segments = dotted_key.split(".")
    start, end, indent = 0, len(lines), 0
    for depth, seg in enumerate(segments):
        prefix = " " * indent + seg + ":"
        found = None
        for i in range(start, end):
            if lines[i] == prefix or lines[i].startswith(prefix + " "):
                found = i
                break
        if found is None:
            block, pad = [], indent
            for offset, tail_seg in enumerate(segments[depth:]):
                tail = f" {value}" if depth + offset == len(segments) - 1 else ""
                block.append(" " * pad + tail_seg + ":" + tail)
                pad += 2
            insert_at = _last_content_line(lines, start, end)
            lines[insert_at:insert_at] = block
            return "\n".join(lines) + "\n"
        if depth == len(segments) - 1:
            rest = lines[found][len(prefix) :]
            hash_pos = rest.find(" #")
            comment = rest[hash_pos:] if hash_pos != -1 else ""
            lines[found] = f"{prefix} {value}{comment}"
            return "\n".join(lines) + "\n"
        # Descend: the section body runs until the first non-blank line at parent indent
        # or shallower (top-level comment banners end a section on purpose — appended keys
        # must never land under the next section's banner).
        sec_end = found + 1
        while sec_end < end:
            line = lines[sec_end]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            sec_end += 1
        start, end, indent = found + 1, sec_end, indent + 2
    return "\n".join(lines) + "\n"


def _last_content_line(lines: list[str], start: int, end: int) -> int:
    """The insertion index just past the last non-blank line in ``[start, end)``."""
    i = end
    while i > start and not lines[i - 1].strip():
        i -= 1
    return i


# ─────────────────────────────────────────────────────────────────────────────
# Probes (each patchable in tests; none imports a heavy stack at module load)
# ─────────────────────────────────────────────────────────────────────────────
def missing_extras() -> list[str]:
    """The optional extras whose representative module is not importable."""
    import importlib.util

    importlib.invalidate_caches()  # a just-finished `uv sync` must be visible immediately
    return [x for x, mod in EXTRA_MODULES.items() if importlib.util.find_spec(mod) is None]


def probe_ollama(base_url: str = OLLAMA_URL) -> list[str] | None:
    """Model tags served by a local Ollama, or ``None`` when nothing answers."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2.0) as resp:
            payload = json.load(resp)
    except Exception:  # noqa: BLE001 — any failure means "no local backend here"
        return None
    return [m["name"] for m in payload.get("models", []) if m.get("name")]


def verify_databento(key: str) -> tuple[bool, str]:
    """Whether ``key`` authenticates against DataBento, via the free metadata endpoint.

    Soft-passes when the ``data`` extra isn't installed — there is nothing to call with,
    and refusing to save the key over that would be backwards.
    """
    try:
        import databento
    except ImportError:
        return True, "saved (install the data extra, then `noctis setup --check` verifies it)"
    try:
        databento.Historical(key=key).metadata.list_datasets()
    except Exception as exc:  # noqa: BLE001 — auth/transport failures all mean "not usable"
        return False, f"rejected by DataBento: {exc}"
    return True, "verified against the DataBento metadata API"


def verify_llm(settings) -> tuple[bool, str]:
    """Prove the configured research model actually answers, with one tiny completion.

    ``client_status`` says a client *can be built*; this says the endpoint, key, and model
    id are real — the difference between "configured" and "works", which is exactly what an
    operator needs to know before an unattended overnight run. Costs a handful of tokens on
    a hosted provider, nothing on a local backend.
    """
    from noctis.research import client_status
    from noctis.research.llm import build_llm_client

    status = client_status(settings)
    if not status.ok:
        return False, f"{status.model}: {status.reason}"
    client = build_llm_client(settings)
    if client is None:
        return False, f"{status.model}: client construction failed"
    import litellm

    litellm.suppress_debug_info = True  # our one red failure line, not litellm's banner spam
    started = time.monotonic()
    try:
        client.complete(
            system="Connectivity check.",
            tools=[],
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=16,
        )
    except Exception as exc:  # noqa: BLE001 — surface any transport/auth failure as text
        return False, f"{status.model}: {exc}"
    return True, f"{status.model} answered in {time.monotonic() - started:.1f}s"


def run_uv_sync(root: Path) -> bool:
    """Run ``uv sync --all-extras`` in ``root``, streaming its output; True on success."""
    import subprocess

    return subprocess.run(["uv", "sync", "--all-extras"], cwd=root).returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# The wizard
# ─────────────────────────────────────────────────────────────────────────────
def run_setup(
    *,
    config_path: str | None = None,
    check_only: bool = False,
    databento_key: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    assume_yes: bool = False,
) -> int:
    """Drive the four setup steps; returns the process exit code.

    Interactive by default; ``assume_yes`` takes every default and never prompts (the
    unattended best-effort mode), and the value flags pre-answer individual prompts.
    ``check_only`` runs every probe read-only and exits 1 when anything is missing.
    """
    from noctis.bootstrap import scaffold_init
    from noctis.config import load_settings
    from noctis.config.settings import _yaml_path

    settings = load_settings(config_path=config_path)
    config_file = _yaml_path()
    root = config_file.parent
    issues: list[str] = []
    interactive = not assume_yes and not check_only

    typer.echo("── Noctis setup ─────────────────────────────────────")

    # [1/4] Local files — the idempotent scaffold `noctis init` also runs.
    _step(1, "Local files")
    if check_only:
        for label, path in (
            ("config.yaml", config_file),
            (".env", root / ".env"),
            ("mandate/MANDATE.md", root / "mandate" / "MANDATE.md"),
            ("workspace/", Path(settings.workspace_dir)),
        ):
            present = path.exists()
            _report(f"{label:<20} {'present' if present else 'MISSING'}", ok=present)
            if not present:
                issues.append(f"{label} missing — run `noctis setup` to scaffold it")
    else:
        for line in scaffold_init(settings):
            typer.echo(f"  {line}")
        settings = load_settings(config_path=config_path)  # the scaffolded config is live now

    # [2/4] Optional components — the seams a full install fills.
    _step(2, "Components")
    missing = missing_extras()
    if not missing:
        _report("all optional components installed", ok=True)
    else:
        _report(f"missing: {', '.join(missing)}", ok=False)
        if check_only:
            issues.append(f"extras not installed ({', '.join(missing)}) — `uv sync --all-extras`")
        elif _can_uv_sync(root) and (
            assume_yes or typer.confirm("  Install everything now (uv sync --all-extras)?", True)
        ):
            if run_uv_sync(root):
                missing = missing_extras()
                still = f"still missing: {', '.join(missing)}" if missing else "installed"
                _report(still, ok=not missing)
            else:
                _report("`uv sync --all-extras` failed — fix the error above and re-run", ok=False)
        else:
            typer.echo("  Install later with: uv sync --all-extras   (run from the repo root)")

    # [3/4] Market data — the research lake ingests from DataBento.
    _step(3, "Market data (DataBento)")
    key = databento_key or settings.databento_api_key or ""
    if not key and interactive:
        typer.echo(
            "  The research/backtest lake ingests history from DataBento; a free signup\n"
            "  credit more than covers the default backfill (https://databento.com)."
        )
        key = typer.prompt(
            "  DataBento API key (typing hidden; Enter to skip)",
            default="",
            show_default=False,
            hide_input=True,
        ).strip()
        if key:
            typer.echo(f"  received {_mask_key(key)}")
    if key:
        if not check_only:
            set_env_key(root / ".env", "DATABENTO_API_KEY", key)
            os.environ["DATABENTO_API_KEY"] = key  # visible to this process immediately
        ok, detail = verify_databento(key)
        _report(f"DATABENTO_API_KEY {detail}", ok=ok)
        if not ok:
            issues.append("DATABENTO_API_KEY is set but was rejected — check the key in .env")
    else:
        _report("no DATABENTO_API_KEY — the lake cannot backfill history", ok=False)
        issues.append("no DATABENTO_API_KEY in .env (research needs catalog data)")

    # [4/4] The LLM — configure if needed, then prove it answers.
    _step(4, "The LLM")
    llm_line = _configure_llm(
        settings=settings,
        config_file=config_file,
        root=root,
        model=model,
        api_key=api_key,
        interactive=interactive,
        check_only=check_only,
        config_path=config_path,
        issues=issues,
    )

    typer.echo("\n── Summary ──────────────────────────────────────────")
    typer.echo(f"  mode      {settings.mode} (paper-only; live needs both safety gates)")
    typer.echo(f"  llm       {llm_line}")
    data_line = "DATABENTO_API_KEY set" if key else "no key — add DATABENTO_API_KEY to .env"
    if key and settings.data.auto_backfill:
        data_line += (
            f"; first `run` backfills {settings.data.history_days} days for "
            f"{len(settings.universe)} symbols (budget ${settings.data.budget_usd:g})"
        )
    typer.echo(f"  data      {data_line}")
    if check_only:
        if issues:
            typer.secho(f"\n{len(issues)} issue(s):", fg=typer.colors.YELLOW)
            for issue in issues:
                typer.echo(f"  - {issue}")
            return 1
        typer.secho("\nEverything checks out.", fg=typer.colors.GREEN)
        return 0
    typer.echo("\n  Next:  uv run python -m noctis run -v")
    typer.echo("         uv run python -m noctis research -v   # one observable session first")
    return 0


def _configure_llm(
    *,
    settings,
    config_file: Path,
    root: Path,
    model: str | None,
    api_key: str | None,
    interactive: bool,
    check_only: bool,
    config_path: str | None,
    issues: list[str],
) -> str:
    """The LLM step: apply flags or walk the menu, then live-verify. Returns the summary line."""
    from noctis.config import load_settings
    from noctis.research import client_status

    configured = False
    if model and not check_only:
        _apply_model(config_file, root, model, api_key, settings)
        configured = True
    elif not client_status(settings).ok and interactive:
        configured = _llm_menu(config_file, root, settings)

    if configured:
        settings = load_settings(config_path=config_path)

    status = client_status(settings)
    if not status.ok:
        _report(f"no working LLM ({status.reason})", ok=False)
        typer.echo("  Research falls back to the legacy (no-LLM) optimizer loop until one is set.")
        issues.append(f"no working LLM for {status.model!r}: {status.reason}")
        return f"none — legacy research loop ({status.reason})"
    ok, detail = verify_llm(settings)
    _report(f"{'verified — ' if ok else 'FAILED — '}{detail}", ok=ok)
    if not ok:
        issues.append(f"LLM configured but not answering: {detail}")
        return f"{status.model} — CONFIGURED BUT NOT ANSWERING"
    return f"{status.model} — verified"


def _llm_menu(config_file: Path, root: Path, settings) -> bool:
    """The interactive provider menu; returns True when a model was configured."""
    local_tags = probe_ollama()
    typer.echo("  Noctis needs an LLM to research (any of these):")
    typer.echo("    1. Anthropic (hosted — paste an API key)")
    typer.echo("    2. OpenAI (hosted — paste an API key)")
    detected = f" — server detected, {len(local_tags)} model(s)" if local_tags else ""
    typer.echo(f"    3. Local Ollama / noctis-ollama ($0/token){detected}")
    typer.echo("    4. Skip for now (research runs the legacy no-LLM loop)")
    choice = typer.prompt("  Choose [1-4]", default="3" if local_tags else "1")
    if choice in ("1", "2"):
        provider = "anthropic" if choice == "1" else "openai"
        entered = typer.prompt(
            f"  {_PROVIDER_ENV[provider]} (typing hidden)",
            hide_input=True,
            default="",
            show_default=False,
        ).strip()
        if not entered:
            typer.echo("  No key entered — skipping.")
            return False
        typer.echo(f"  received {_mask_key(entered)}")
        chosen = typer.prompt("  Model", default=_HOSTED_DEFAULT_MODEL[provider])
        _apply_model(config_file, root, chosen, entered, settings)
        return True
    if choice == "3":
        if not local_tags:
            typer.echo(
                "  No Ollama server at localhost:11434. Stand one up with noctis-ollama\n"
                "  (installs Ollama, provisions verified agent-ready models):\n"
                f"      git clone {NOCTIS_OLLAMA_REPO} && cd noctis-ollama && ./setup.sh\n"
                "  then re-run `noctis setup` — it will find the server and finish the wiring."
            )
            return False
        ordered = sorted(local_tags, key=lambda t: not t.startswith("noctis-"))
        typer.echo("  Models on the local server:")
        for i, t in enumerate(ordered, 1):
            mark = "  (agent-ready)" if t.startswith("noctis-") else ""
            typer.echo(f"    {i}. {t}{mark}")
        picked = typer.prompt(f"  Choose [1-{len(ordered)}] or type a tag", default="1").strip()
        # Anything that isn't a listed number is taken as a literal tag — the escape hatch
        # for a model the server doesn't serve yet (e.g. one still pulling).
        in_range = picked.isdigit() and 1 <= int(picked) <= len(ordered)
        tag = ordered[int(picked) - 1] if in_range else picked
        _apply_model(config_file, root, f"ollama_chat/{tag}", None, settings)
        return True
    return False


def _apply_model(config_file: Path, root: Path, model: str, api_key: str | None, settings) -> None:
    """Write the chosen model into ``config.yaml`` (and its key into ``.env``) surgically."""
    from noctis.research import provider_of

    pairs = [("research.model", model)]
    # A local backend usually needs the output cap the noctis-ollama contract test assumes;
    # never clobber a value the operator already pinned.
    if model.startswith("ollama") and settings.research.agent.max_tokens is None:
        pairs.append(("research.agent.max_tokens", "4096"))
    text = config_file.read_text() if config_file.is_file() else ""
    for dotted, value in pairs:
        text = set_config_value(text, dotted, value)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(text)
    typer.echo(f"  wrote research.model: {model} → {config_file}")
    if api_key:
        provider = provider_of(model)
        env_name = _PROVIDER_ENV.get(provider, f"{provider.upper()}_API_KEY")
        set_env_key(root / ".env", env_name, api_key)
        os.environ[env_name] = api_key  # visible to this process immediately
        typer.echo(f"  wrote {env_name} → {root / '.env'}")


def _can_uv_sync(root: Path) -> bool:
    """Whether the wizard can run ``uv sync`` here (uv on PATH, a project to sync)."""
    import shutil

    return shutil.which("uv") is not None and (root / "pyproject.toml").is_file()


def _step(n: int, title: str) -> None:
    typer.echo(f"\n[{n}/4] {title}")


def _mask_key(value: str) -> str:
    """Enough of a hidden-input paste to confirm it landed, without echoing the secret."""
    if len(value) <= 8:
        return f"{len(value)} chars"
    return f"{value[:4]}…{value[-4:]} ({len(value)} chars)"


def _report(message: str, *, ok: bool) -> None:
    typer.secho(f"  {'✓' if ok else '✗'} {message}", fg=None if ok else typer.colors.YELLOW)
