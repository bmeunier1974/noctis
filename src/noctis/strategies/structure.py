"""Structural lint — AST coherence checks the write gate runs on raw source, before import.

The behavioural gate (import, smoke replay, scenarios, parity, the Tier-1 invariants) proves
what a file *does*; it is blind to a file that is incoherent as *text*. A class body that
defines ``on_bar`` twice imports cleanly, replays cleanly and passes every scenario — the
second definition silently overrides the first, so half the authored thesis never runs. That is
a defect no behavioural check can see, and it crowned a champion once (#158).

:func:`check_structure` is the whole surface: raw source in, the first violation out as one
line (or ``None``). It parses once and runs the ordered :data:`_CHECKS` — small pure functions
over the parsed tree — so a new check is a function plus one tuple entry. Stdlib ``ast`` only,
deliberately: this runs inside the validation subprocess, before the candidate is imported, and
must cost nothing and depend on nothing.
"""

from __future__ import annotations

import ast

# A definition statement, in either flavour — ``async def`` overrides its predecessor exactly
# as ``def`` does.
_Def = ast.FunctionDef | ast.AsyncFunctionDef


def _scopes(tree: ast.Module) -> list[tuple[str, list[ast.stmt]]]:
    """Every body where a definition can shadow a sibling: module top level and class bodies.

    Class bodies are collected from the whole tree, not just the top level, because the strategy
    contract nests ``class Params`` inside the strategy class. Function bodies are out of scope
    for now: a local helper redefined inside one function is a smaller, more local mistake.
    """
    scopes: list[tuple[str, list[ast.stmt]]] = [("at module level", tree.body)]
    scopes += [
        ("in class body", node.body) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]
    return scopes


def _check_duplicate_definitions(tree: ast.Module) -> str | None:
    """Refuse a scope that defines the same *undecorated* function name twice.

    Undecorated is the whole discrimination: a decorated redefinition is the language's own
    idiom for extending a name (``@property`` / ``@name.setter``, ``@singledispatch`` registration)
    and both definitions stay reachable. Two bare ``def``s under one name never do — the second
    replaces the first, and nothing at runtime says so.
    """
    violations: list[tuple[int, str]] = []
    for where, body in _scopes(tree):
        seen: set[str] = set()
        for node in body:
            if not isinstance(node, _Def) or node.decorator_list:
                continue
            if node.name in seen:
                violations.append(
                    (
                        node.lineno,
                        f"duplicate definition of {node.name!r} {where} — the second silently "
                        f"overrides the first; keep exactly one",
                    )
                )
            seen.add(node.name)
    return min(violations)[1] if violations else None


# Ordered: the first violation is the one reported, and duplicate definitions come first because
# a shadowed definition makes every later reading of the file misleading.
_CHECKS = (_check_duplicate_definitions,)


def check_structure(source: str) -> str | None:
    """The first structural violation in ``source`` as one line, or ``None`` if it is coherent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Unparsable source is the import step's failure to report, in its canonical wording;
        # inventing a second dialect for it here would only give the REPAIR loop two to match.
        return None
    for check in _CHECKS:
        message = check(tree)
        if message:
            return message
    return None
