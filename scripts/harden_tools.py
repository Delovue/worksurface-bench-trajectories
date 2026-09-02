"""Runtime hardening for DataMind zero-parameter tool schemas.

The bug
-------
Six tools declare themselves as taking no arguments like this::

    input_schema={"type": "object", "properties": {}}

    db_list_tables, kb_list_documents, kb_count, kb_reindex,
    memory_list_profiles, skill_list

In JSON Schema, `additionalProperties` defaults to **true**. So that document
does not say "I accept nothing" — it says "I declare no named parameters, and
extra ones are permitted." Meanwhile the handlers take zero parameters::

    async def _list_tables() -> dict:              # one extra kwarg -> TypeError
        return {"tables": await db.list_tables()}

The published contract and the implementation disagree, which is the actual
defect — independent of model behaviour. A model that sends
`{"_call": "find the table with ..."}` is doing something odd but is *not*
violating the schema it was given.

Observed impact: `db_list_tables` raised
`TypeError: _list_tables() got an unexpected keyword argument '_call'`
in 5 of 6 tool failures across a 24-task run, and again on a WorkSurface-Bench
task where it consumed the step that would have found the answer. It is
intermittent — the same tool was called correctly with `{}` on other runs — so
it injects failures that have nothing to do with task difficulty.

The fix, applied at the `ToolRegistry.add` seam so every provider is covered
without editing the DataMind tree:

1. Add `additionalProperties: false` to any schema that declares no
   properties. This makes the contract state what the handler actually
   enforces, so the model is told the truth up front.
2. Wrap zero-parameter handlers to discard unexpected kwargs, so a model that
   ignores the schema anyway degrades to a correct call rather than a crash.

Only zero-parameter handlers are wrapped. Tools with real parameters keep
strict behaviour on purpose: silently dropping a misspelled argument there
could turn a loud failure into a quietly wrong answer.

Usage — import before building the agent (traj/sitecustomize.py does this):

    import harden_tools; harden_tools.apply()
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any

from datamind.core.tools import ToolRegistry

_PATCH_FLAG = "_datamind_traj_hardened"


def takes_no_params(handler: Any) -> bool:
    """True only if `handler` accepts no caller-supplied arguments at all.

    A `**kwargs` handler is already tolerant of extras, so it needs no wrapper
    and is reported as False.
    """
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    return not signature.parameters


def harden_spec(spec: Any) -> Any:
    """Return a spec whose schema and handler agree about taking no arguments."""
    schema = getattr(spec, "input_schema", None)
    if not isinstance(schema, dict):
        return spec
    # Only touch the "declares nothing" case, and never override an explicit
    # additionalProperties the author already set.
    if schema.get("properties") or "additionalProperties" in schema:
        return spec

    # The dataclass is frozen, but the dict it holds is not — mutating the
    # schema in place needs no replace() and keeps object identity for any
    # caller that already captured this spec.
    schema["additionalProperties"] = False

    handler = getattr(spec, "handler", None)
    if handler is None or not takes_no_params(handler):
        return spec

    @functools.wraps(handler)
    async def tolerant_handler(**kwargs: Any):
        # Contract is zero-arg; anything the model invented is discarded.
        return await handler()

    try:
        return dataclasses.replace(spec, handler=tolerant_handler)
    except Exception:
        # Schema tightening alone still helps if the spec cannot be rebuilt.
        return spec


def apply() -> None:
    """Idempotently patch ToolRegistry.add. `extend` delegates to it."""
    if getattr(ToolRegistry, _PATCH_FLAG, False):
        return

    original_add = ToolRegistry.add

    @functools.wraps(original_add)
    def add(self: ToolRegistry, spec: Any) -> None:
        return original_add(self, harden_spec(spec))

    ToolRegistry.add = add  # type: ignore[method-assign]
    setattr(ToolRegistry, _PATCH_FLAG, True)


__all__ = ["apply", "harden_spec", "takes_no_params"]
