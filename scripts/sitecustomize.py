"""Apply collection-side patches in every Python process.

Python imports `sitecustomize` at interpreter startup when it is found on
sys.path, so putting this directory on PYTHONPATH covers `python -m datamind`,
`python -m benchmark.run`, and `collect.py` alike without editing DataMind.

Only one patch is active now: `harden_tools` reconciles the six
zero-parameter tool schemas with their handlers (see that module for why).

The former `local_embed` shim is retired — DataMind's built-in `huggingface`
provider with BAAI/bge-m3 replaced it once hf-mirror.com became reachable.
"""
try:
    import harden_tools

    harden_tools.apply()
except Exception as exc:  # pragma: no cover - never break interpreter startup
    import sys

    print(f"[sitecustomize] harden_tools NOT applied: {exc!r}", file=sys.stderr)
