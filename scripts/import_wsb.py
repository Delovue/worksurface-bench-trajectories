"""Import WorkSurface-Bench workspaces into DataMind profiles.

WorkSurface-Bench ships each persona workspace already split along the three
surfaces DataMind exposes, so the mapping is structural rather than semantic:

    resources/profiles/<persona>/kb_docs/*.md        -> kb     (DataMind indexes it)
    resources/profiles/<persona>/tables/*.parquet    -> db     (SQLite table per file)
    resources/profiles/<persona>/graph/surface_graph.json -> graph (triplet JSONL)

Three details decide whether `gold_evidence` can still be checked afterwards,
so they are preserved exactly rather than normalised:

1. **Table names.** `gold_evidence[].table` equals the parquet filename minus
   its extension (e.g. `t300__activity_taobaoactivity_followup_sheet__applyjin_du`).
   Renaming or lower-casing would silently break evidence matching.
2. **Graph node ids.** `gold_evidence[].graph_path` is literally
   `[subject, relation, object]`, so each node-link edge becomes one triple with
   ids untouched — including the `::` separator, which differs from the `__`
   used by KB filenames for the same underlying file.
3. **Doc filenames.** `kb_docs` names carry the `t<task>__<file>` prefix that
   joins a document back to its source task; they are copied verbatim.

Node attributes (type/persona/difficulty/filename/ext/task) have nowhere to
live in an edge-shaped triple, so they are folded into each triple's
`properties` as `subject_meta` / `object_meta`. Nothing is invented as a
synthetic edge, which would inflate the graph and confuse `graph_path` lookups.

Usage:
    python import_wsb.py --src .../worksurfacebench --datamind .../DataMind
    python import_wsb.py --src ... --datamind ... --personas backend_developer
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

PROFILE_PREFIX = "wsb_"


def convert_graph(graph_json: Path, out_jsonl: Path) -> dict[str, int]:
    """node-link JSON -> DataMind GraphTriple JSONL (one triple per edge)."""
    graph = json.loads(graph_json.read_text(encoding="utf-8"))
    nodes: dict[str, dict[str, Any]] = {
        str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")
    }

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for edge in graph.get("edges", []) or graph.get("links", []):
            subject = edge.get("from") or edge.get("source")
            obj = edge.get("to") or edge.get("target")
            relation = edge.get("rel") or edge.get("relation")
            if not (subject and obj and relation):
                skipped += 1
                continue

            subject_node = nodes.get(str(subject), {})
            object_node = nodes.get(str(obj), {})
            triple = {
                "subject": str(subject),
                "relation": str(relation),
                "object": str(obj),
                # `type` is the closest thing the source has to an entity kind.
                "subject_type": subject_node.get("type", "entity"),
                "object_type": object_node.get("type", "entity"),
                "source": edge.get("source") or "worksurface-bench",
                "properties": {
                    "subject_meta": {
                        k: v for k, v in subject_node.items() if k != "id"
                    },
                    "object_meta": {
                        k: v for k, v in object_node.items() if k != "id"
                    },
                },
            }
            handle.write(json.dumps(triple, ensure_ascii=False) + "\n")
            written += 1

    return {"nodes": len(nodes), "edges_written": written, "edges_skipped": skipped}


def import_tables(tables_dir: Path, db_path: Path) -> dict[str, Any]:
    """Load every parquet into SQLite, keeping the filename as the table name."""
    import pandas as pd

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    loaded: list[str] = []
    failures: list[dict[str, str]] = []
    total_rows = 0
    try:
        for parquet in sorted(tables_dir.glob("*.parquet")):
            table = parquet.stem  # MUST match gold_evidence[].table
            try:
                frame = pd.read_parquet(parquet)
                # Duplicate column labels are legal in a DataFrame but not in
                # SQL; disambiguate rather than let to_sql fail late.
                if frame.columns.duplicated().any():
                    seen: dict[str, int] = {}
                    renamed = []
                    for column in frame.columns:
                        if column in seen:
                            seen[column] += 1
                            renamed.append(f"{column}_{seen[column]}")
                        else:
                            seen[column] = 0
                            renamed.append(column)
                    frame.columns = renamed
                frame.to_sql(table, connection, if_exists="replace", index=False)
                loaded.append(table)
                total_rows += len(frame)
            except Exception as exc:  # noqa: BLE001 - report, don't abort the batch
                failures.append({"table": table, "error": f"{type(exc).__name__}: {exc}"})
        connection.commit()
    finally:
        connection.close()
    return {"tables": len(loaded), "rows": total_rows, "failures": failures}


def import_kb(kb_dir: Path, dest_dir: Path) -> dict[str, int]:
    """Copy markdown docs verbatim; DataMind's indexer walks this recursively."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for doc in sorted(kb_dir.glob("*.md")):
        shutil.copy2(doc, dest_dir / doc.name)
        copied += 1
    registry = kb_dir / "registry.json"
    if registry.is_file():
        # Kept for provenance; .json is not indexed as a KB document.
        shutil.copy2(registry, dest_dir.parent / "kb_registry.json")
    return {"docs": copied}


def import_persona(persona_dir: Path, datamind_root: Path) -> dict[str, Any]:
    profile = f"{PROFILE_PREFIX}{persona_dir.name}"
    data_dir = datamind_root / "data" / "profiles" / profile
    storage_dir = datamind_root / "storage" / profile

    # Idempotent: a re-run must not merge two imports.
    for stale in (data_dir, storage_dir):
        if stale.exists():
            shutil.rmtree(stale)
    data_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"profile": profile}

    kb_dir = persona_dir / "kb_docs"
    if kb_dir.is_dir():
        report["kb"] = import_kb(kb_dir, data_dir / "kb_docs")

    graph_json = persona_dir / "graph" / "surface_graph.json"
    if graph_json.is_file():
        report["graph"] = convert_graph(graph_json, data_dir / "triplets" / "surface_graph.jsonl")

    tables_dir = persona_dir / "tables"
    if tables_dir.is_dir():
        report["db"] = import_tables(tables_dir, storage_dir / "demo.db")
        registry = tables_dir / "registry.json"
        if registry.is_file():
            shutil.copy2(registry, data_dir / "tables_registry.json")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="worksurfacebench root")
    parser.add_argument("--datamind", required=True, help="DataMind repo root")
    parser.add_argument("--personas", help="comma-separated subset (default: all)")
    args = parser.parse_args()

    profiles_dir = Path(args.src) / "resources" / "profiles"
    if not profiles_dir.is_dir():
        raise SystemExit(f"[ERROR] not found: {profiles_dir}")
    datamind_root = Path(args.datamind)

    wanted = set(filter(None, (args.personas or "").split(","))) or None
    personas = [
        d for d in sorted(profiles_dir.iterdir())
        if d.is_dir() and (wanted is None or d.name in wanted)
    ]
    if not personas:
        raise SystemExit("[ERROR] no matching personas")

    reports = []
    for persona_dir in personas:
        report = import_persona(persona_dir, datamind_root)
        reports.append(report)
        kb = report.get("kb", {})
        graph = report.get("graph", {})
        db = report.get("db", {})
        print(
            f"  {report['profile']:34s} "
            f"docs={kb.get('docs', 0):4d} "
            f"nodes={graph.get('nodes', 0):4d} edges={graph.get('edges_written', 0):4d} "
            f"tables={db.get('tables', 0):4d} rows={db.get('rows', 0):7d}"
        )
        for failure in db.get("failures", []):
            print(f"      !! table {failure['table']}: {failure['error'][:120]}")

    print()
    print(f"imported {len(reports)} profiles into {datamind_root}")
    print("next: index KB per profile, e.g.")
    print(f"  DATAMIND__DATA__PROFILE={reports[0]['profile']} python -m datamind ingest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
