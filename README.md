# WorkSurface-Bench trajectories on DataMind

1,151 agent trajectories — one per WorkSurface-Bench task — collected by running the
benchmark's five persona workspaces through [DataMind](https://github.com/OpenDCAI/DataMind)
as the agent harness.

The point of the collection is the **trajectory**, not a score: each record keeps the
full message history, every tool call, the untruncated tool output, and the retrieved
evidence, so the agent's decision process can be studied after the fact.

## What is in a record

One JSON object per line. Beyond the answer, each record carries the parts that a
scoring-oriented runner normally discards:

| field | what it holds |
|---|---|
| `answer` | **the model's final response** — what the agent actually answered (present and non-empty on all 1,151 records). Compare against `reference_answer` for the gold. |
| `question` | the task prompt as given to the agent |
| `history` | full message list — every assistant text block (the model's own reasoning between tool calls), every `tool_use`, every `tool_result` fed back |
| `tool_trace` | per-call metadata: tool name, surface, access level, input, `is_error`, latency, result size |
| `evidence` | retrieved items with `surface`, `source_id`, `locator`, `content`, score |
| `raw_tool_results` | **untruncated** tool returns; `history` holds the truncated version and `tool_trace` only a char count, so this is the only complete copy |
| `nested_model_usage` | model calls made *inside* a tool (NL2SQL), which the outer agent loop does not see |
| `usage`, `iterations`, `stop_reason`, `latency_s` | run accounting |
| `attempts` | retry count — **only present on the 67 re-collected records** (retry support was added after the first pass); absent means a single attempt |
| `reference_answer`, `metadata` | the task's gold answer, `gold_tools`, `gold_evidence`, `required_surfaces`, `answer_type`, `difficulty`, `task_type` |

`data/` holds one file per persona: `backend_developer` (181), `product_manager` (39),
`researcher` (44), `logistics_manager` (595), `operations_manager` (292).

The two answer fields are easy to confuse, so concretely:

```python
import json
rec = json.loads(open("data/wsb_backend_developer.jsonl").readline())

rec["answer"]            # what the model produced — free-form prose, often
                         # a Markdown table with its reasoning and sources
rec["reference_answer"]  # the benchmark gold, e.g. "_activity_..._sheet.xlsx; 9"
```

`answer` is unconstrained: the model was not asked for a bare value, so a correct
response typically embeds the gold fragments inside an explanation. That is why the
`substring_hit_rate` below is a fragment check rather than an equality test.

## Collection setup

| | |
|---|---|
| Harness | DataMind v0.3.2, `native` agent loop |
| Model | `claude-sonnet-4-6`, max 14 turns / 28 tool calls per task |
| Surfaces enabled | `kb`, `db`, `graph`, `skills` |
| Embedding | `BAAI/bge-m3` (1024-dim, multilingual) via DataMind's built-in `huggingface` provider |
| Retrieval | hybrid (dense + BM25, RRF fusion), top-k 6 |
| Outcome | 1,151/1,151 completed, **0 task-level failures**; 4,658 tool calls at a 3.3% error rate; 8,840 messages; 9.66M input / 688K output tokens; 10.3 h wall clock |

WorkSurface-Bench's three surfaces map onto DataMind's structurally, which is what makes
this substitution possible: `kb_docs/*.md` → kb, `tables/*.parquet` → db (SQLite, one
table per parquet), `graph/surface_graph.json` → graph (node-link JSON converted to
triples). `scripts/import_wsb.py` does the conversion; see its docstring for the three
identifiers that must survive it verbatim, or `gold_evidence` stops matching.

## Main observation: multi-surface routing collapses to a single surface

Of the 600 tasks whose `required_surfaces` include `graph`, only 202 actually used it —
**66% bypassed a surface the task was labelled as needing.** Split by task type:

| task type | graph required | bypassed | rate |
|---|---|---|---|
| `cross_surface` | 403 | 316 | **78%** |
| `graph_only` | 171 | 61 | **36%** |

When graph is the *only* surface a task needs, the agent usually reaches for it. When the
task needs graph **plus** another surface, it almost always drops graph and answers from
the other one alone. So this is not an inability to use the graph — every graph call in
the corpus succeeded, returning real data — it is that **whenever a single-surface
shortcut exists, the agent takes it.**

The shortcut is built into the harness rather than introduced by our import. The official
`runner/tools.py::table_list()` attaches `source_file` to every table, so "which file
contains which table" — the thing the graph's `task_requires_file` edges exist to answer —
comes back from one table call.

This is consistent with how the benchmark scores routing: in `scoring/route_evidence.py`,
Route is an **F1 over chosen-vs-needed surfaces**, and Answer / Route / Evidence are
independent scores. The benchmark *measures* surface routing rather than *enforcing* it.
The human-audited "surface necessity" screen rules out leakage in the task statement,
which is a different property from being unbypassable at the tool layer.

## Caveats

Read these before quoting any number.

- **`substring_hit_rate` (0.573) is not an official score.** It checks whether every
  comma/semicolon-separated fragment of the gold answer appears somewhere in the response.
  It is lenient about extra text and strict about formatting. Use the official scorers in
  WorkSurface-Bench's `scoring/` for comparable figures.
- **Bypassing graph barely changed answer accuracy** (32% with graph vs 29% without). The
  3-point gap is too small to support a claim that bypassing costs correctness. The
  failure is in the routing, not the answer.
- **26 `backend_developer` records lack `task_type`** (the early extraction script omitted
  the field), so they appear as `unlabeled` in the breakdown and the task-type denominators
  sum to 574, not 600.
- **`surfaces_used` over-reports.** DataMind appends to it from the tool trace even when a
  call errored, so it means "attempted" rather than "returned data". The bypass numbers
  here are computed from it, which is the conservative direction for this particular claim
  (it can only *under*-count bypasses). For grounding checks use `evidence[].surface`
  instead — and note that `utility`-access tools such as `calculator` never emit evidence
  by design.
- **One table was not imported.** `t45__t_6_..._chart_linkage_selected_analysis__sheet1`
  has 16,377 columns, past SQLite's 2,000-column ceiling. No task references it.
- Answers are in mixed Chinese and English: the model was prompted in the tasks' original
  English but often reasons and replies in Chinese.

## Reproducing

```bash
# 1. import the benchmark workspaces into DataMind profiles
python scripts/import_wsb.py --src /path/to/worksurfacebench --datamind /path/to/DataMind

# 2. build the KB index per profile (see env.sh for configuration)
source scripts/env.sh
DATAMIND__DATA__PROFILE=wsb_backend_developer python -m datamind ingest

# 3. collect trajectories
DATAMIND__DATA__PROFILE=wsb_backend_developer python scripts/collect.py \
    --questions tasks.jsonl --output out/trajectories.jsonl \
    --surfaces kb,db,graph,skills --concurrency 4
```

Two things in `scripts/` are worth knowing about independently of this dataset:

`harden_tools.py` fixes a real DataMind defect. Seven tools declare
`{"type": "object", "properties": {}}` without `additionalProperties: false`. In JSON
Schema that permits extra arguments, but the handlers take none, so a model that sends
`{"_call": "..."}` — which the published schema allows — crashes them with a `TypeError`.
The patch reconciles schema and handler at the `ToolRegistry.add` seam without editing
DataMind. Before it, this accounted for 5 of 6 tool failures in a 24-task run and once
consumed the step that would have found the answer.

`collect.py` retries only transient gateway errors (502/503/504/429). Contract violations
and tool bugs are recorded as-is, because those are signal about the run.

Re-indexing gotcha: `storage/<profile>/ingest_state.json` is a dedupe ledger keyed on the
*source* checksum and does not track the embedding model. After switching embeddings,
`ingest` reports `unchanged` and writes zero vectors. Delete `ingest_state.json`,
`kb_index_manifest.json`, and `chroma/` together to force a rebuild.
(arXiv:2605.03596); the WorkSurface-Bench report is arXiv:2607.25765.

The harness is [DataMind](https://github.com/OpenDCAI/DataMind) by OpenDCAI.

Trajectories in `data/` are derived from CC-BY-4.0 material and are shared under the same
terms. Please cite WorkSurface-Bench if you use them.
