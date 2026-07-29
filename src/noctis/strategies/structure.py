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


def _self_attr(node: ast.expr) -> str | None:
    """The attribute name if ``node`` is ``self.<attr>``, else ``None``."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.attr if node.value.id == "self" else None
    return None


def _methods(cls: ast.ClassDef) -> list[_Def]:
    """The methods of ``cls`` itself — a nested ``class Params`` owns its own ``self``."""
    return [node for node in cls.body if isinstance(node, _Def)]


def _updated_attrs(method: _Def) -> set[str]:
    """Attributes ``method`` moves by any means other than a plain assignment.

    Aug-assignment, a method call (``self._closes.append(...)`` — the deque idiom every seed
    uses), an item/slice store or a ``del``: each one changes the object under the name without
    ever appearing as ``self.<attr> = ...``, so each one is proof of life.
    """
    updated: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.AugAssign):
            name = _self_attr(node.target)
        elif isinstance(node, ast.Call):
            name = _self_attr(node.func.value) if isinstance(node.func, ast.Attribute) else None
        elif isinstance(node, ast.Subscript | ast.Starred) and isinstance(
            node.ctx, ast.Store | ast.Del
        ):
            name = _self_attr(node.value)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Del):
            name = _self_attr(node)
        else:
            continue
        if name:
            updated.add(name)
    return updated


def _assigned_attrs(method: _Def) -> dict[str, bool]:
    """Attributes ``method`` plainly assigns, mapped to "the value is a bare literal".

    A literal is :class:`ast.Constant` and nothing else. A call, an attribute read, an
    arithmetic expression — anything derived from ``Params`` — is config, which is legitimately
    assigned once and read forever; only a constant pin can make a branch unreachable.
    """
    assigned: dict[str, bool] = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        literal = isinstance(value, ast.Constant)
        for target in targets:
            name = _self_attr(target)
            if name:
                assigned[name] = assigned.get(name, True) and literal
    return assigned


def _read_attrs(method: _Def) -> dict[str, int]:
    """Attributes ``method`` reads, each mapped to the line of its first read."""
    reads: dict[str, int] = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            name = _self_attr(node)
            if name and name not in reads:
                reads[name] = node.lineno
    return reads


def _check_dead_state(tree: ast.Module) -> str | None:
    """Refuse an ``on_bar`` that decides on state pinned to a literal in ``on_start``.

    The champion that motivated this (#158) branched on a ``self._pos`` set to ``0`` in
    ``on_start`` and never touched again: it imports, replays and passes every scenario, because
    the branch simply never fires — half the authored thesis is unreachable and nothing says so.
    The condition is deliberately narrow, so a legitimate file is never caught: the attribute
    must be *read by* ``on_bar``, every assignment to it in the class must be a literal, and all
    of those assignments must live in ``on_start``. An assignment anywhere else (``on_bar``
    included), a derived value, or any in-place update clears it.
    """
    violations: list[tuple[int, str]] = []
    for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        methods = _methods(cls)
        reads: dict[str, int] = {}
        for method in methods:
            if method.name == "on_bar":
                reads |= _read_attrs(method)
        if not reads:
            continue
        pinned: set[str] = set()  # literal-assigned by on_start
        alive: set[str] = set()  # updated in place, or assigned anything else, anywhere
        for method in methods:
            alive |= _updated_attrs(method)
            for name, literal in _assigned_attrs(method).items():
                (pinned if literal and method.name == "on_start" else alive).add(name)
        dead = pinned - alive
        violations += [
            (
                line,
                f"dead state {name!r} in class {cls.name!r} — 'on_bar' reads it but nothing "
                f"ever changes it from the literal 'on_start' assigns, so the branches that "
                f"test it are unreachable; update it or drop them",
            )
            for name, line in reads.items()
            if name in dead
        ]
    return min(violations)[1] if violations else None


# Ordered: the first violation is the one reported, and duplicate definitions come first because
# a shadowed definition makes every later reading of the file misleading — including this one's,
# which would report on a body that never runs.
_CHECKS = (_check_duplicate_definitions, _check_dead_state)


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
