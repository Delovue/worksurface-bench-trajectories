"""Lossless trajectory collector for DataMind.

Why this exists instead of `benchmark/run.py`
---------------------------------------------
The shipped runner is built for *evaluation*: it keeps the final answer plus
summary metadata and drops the rest. For trajectory collection the dropped
parts are the point. Specifically, `benchmark/run.py::_run_one` (~L215-236)
persists only a subset of what `run_turn` returns, losing:

  * `history`   — the full message list: every assistant text block (the
                  model's own reasoning and plans between tool calls), every
                  `tool_use` block, and every `tool_result` fed back in.
                  This is the richest signal in a trajectory.
  * `receipts`  — StoreAgent write receipts.
  * `contract_valid` / `contract_repair_attempted` — whether a formatting
                  repair round happened.

And because the runner catches exceptions around `system.query()`, a
`FinalAnswerContractError` throws the trajectory away entirely: `result`
stays None, so the output row has an empty `tool_trace`. Observed directly —
a 3-way ablation where all three runs (including the one that computed the
right answer) produced empty traces.

This collector additionally binds a `RequestContext`, which captures one more
thing no caller normally sees: `loop_native` writes *untruncated* tool results
into `ctx.extra["raw_tool_results"]` (loop_native.py ~L752). The `tool_trace`
only carries a `result_size_chars` count, and `history` holds the version
truncated to `max_tool_result_chars` — so the raw capture is the only place
the full retrieved payload survives.

Contract failures are recorded as data, not lost: the trajectory is written
with whatever was captured plus the contract error, so a strict
`final_contract` can be used for answer checking without sacrificing the run.

Usage:
    python collect.py --questions tasks.jsonl --output out/traj.jsonl
    python collect.py --questions t.jsonl --output out/abl.jsonl --surfaces graph,db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from datamind import __version__
from datamind.agent import build_datamind
from datamind.config import Settings
from datamind.core.context import RequestContext
from datamind.core.logging import bind_context


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not str(item.get("question", "")).strip():
            raise ValueError(f"line {lineno}: empty question")
        item.setdefault("task_id", str(lineno))
        task_id = str(item["task_id"])
        if task_id in seen:
            raise ValueError(f"line {lineno}: duplicate task_id {task_id!r}")
        seen.add(task_id)
        tasks.append(item)
    return tasks


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one record durably (fsync) so a crash keeps prior rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


# Gateway-side hiccups that produce a 0-step trajectory: pure waste, worth
# retrying. A contract violation or a tool bug is NOT here — those are real
# signal about the run and must be recorded as-is.
_TRANSIENT = ("InternalServerError", "APIConnectionError", "APITimeoutError",
              "RateLimitError", "ServiceUnavailableError", "APIStatusError")


def _is_transient(failure: dict[str, Any] | None) -> bool:
    if not failure:
        return False
    if failure.get("type") in _TRANSIENT:
        return True
    message = str(failure.get("message", ""))
    return any(code in message for code in ("502", "503", "504", "429"))


async def collect_one(
    task: dict[str, Any],
    *,
    system: Any,
    run_id: str,
    surfaces: str,
    profile: str,
    semaphore: asyncio.Semaphore,
    max_attempts: int = 3,
) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        result: dict[str, Any] | None = None
        failure: dict[str, Any] | None = None
        raw_tool_results: list[Any] = []
        nested_model_usage: list[Any] = []
        attempts = 0

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            # Fresh context per attempt so a retry does not inherit the
            # partial tool results of the failed one.
            ctx = RequestContext.new(profile=profile)
            failure = None
            with bind_context(ctx):
                try:
                    result = await system.query(
                        str(task["question"]),
                        final_contract=task.get("final_contract"),
                    )
                except Exception as exc:  # noqa: BLE001 — failures are data here
                    failure = {
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                        "stop_reason": getattr(exc, "stop_reason", None),
                    }
                # Read ctx BEFORE leaving the binding: populated even when the
                # call raised, which is how a contract failure keeps its trace.
                raw_tool_results = list(ctx.extra.get("raw_tool_results", []))
                nested_model_usage = list(ctx.extra.get("nested_model_usage", []))

            if not _is_transient(failure) or attempt == max_attempts:
                break
            await asyncio.sleep(min(2 ** attempt, 30))

        result = result or {}
        row = {
            "run_id": run_id,
            "task_id": str(task["task_id"]),
            "trace_id": ctx.trace_id,
            "session_id": ctx.session_id,
            "question": task["question"],
            "surfaces_enabled": surfaces,
            "latency_s": round(time.perf_counter() - started, 3),
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # >1 means a transient gateway error was retried; the kept trace is
            # from the final attempt.
            "attempts": attempts,
            # ---- everything run_turn returns, nothing dropped -------------
            "answer": result.get("answer", ""),
            "history": result.get("history", []),
            "iterations": result.get("iterations"),
            "stop_reason": result.get("stop_reason"),
            "usage": result.get("usage", {}),
            "tool_trace": result.get("tool_trace", []),
            "evidence": result.get("evidence", []),
            "receipts": result.get("receipts", []),
            "surfaces_used": result.get("surfaces_used", []),
            "contract_valid": result.get("contract_valid"),
            "contract_repair_attempted": result.get("contract_repair_attempted"),
            # ---- only visible via the bound context ------------------------
            "raw_tool_results": raw_tool_results,
            "nested_model_usage": nested_model_usage,
            # ---- failure recorded alongside, never instead of, the trace ---
            "failure": failure,
        }
        for passthrough in ("final_contract", "reference_answer", "metadata"):
            if passthrough in task:
                row[passthrough] = task[passthrough]
        return row


async def _main(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.questions)
    output = Path(args.output)
    if output.exists() and not args.append:
        raise FileExistsError(f"{output} exists; pass --append or use a new path")

    surfaces = set(filter(None, (args.surfaces or "").split(","))) or None
    settings = Settings()
    system = await build_datamind(settings, enable=surfaces)
    run_id = args.run_id or (
        f"traj-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    )
    surfaces_label = ",".join(sorted(surfaces)) if surfaces else "default"

    try:
        await system.warmup()
        semaphore = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()
        done = 0

        async def run(task: dict[str, Any]) -> None:
            nonlocal done
            row = await collect_one(
                task, system=system, run_id=run_id,
                surfaces=surfaces_label, profile=settings.data.profile,
                semaphore=semaphore,
            )
            async with lock:
                _append_jsonl(output, row)
                done += 1
                status = "FAIL" if row["failure"] else "ok"
                print(
                    f"  [{done}/{len(tasks)}] {row['task_id']:10s} {status:4s} "
                    f"steps={len(row['tool_trace']):2d} "
                    f"msgs={len(row['history']):2d} "
                    f"surfaces={'+'.join(row['surfaces_used']) or '-'}",
                    flush=True,
                )

        await asyncio.gather(*(run(task) for task in tasks))
        print(f"\nrun_id={run_id}  surfaces={surfaces_label}  artifact={output}")
        print(f"datamind={__version__}  profile={settings.data.profile}"
              f"  model={settings.llm.model}")
        return 0
    finally:
        await system.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lossless DataMind trajectory collector")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--surfaces", help="comma-separated surfaces to enable (ablation)")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
