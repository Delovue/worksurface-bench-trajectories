# Source this before running seed / collection:  source traj/env.sh
#
# Credentials are *referenced* from the ambient ANTHROPIC_* vars that already
# exist in this environment — deliberately not copied into a file on disk.

set -a

# ---- LLM: the gateway speaks Anthropic Messages (verified HTTP 200) --------
DATAMIND__LLM__API_BASE="${ANTHROPIC_BASE_URL}"
DATAMIND__LLM__API_KEY="${ANTHROPIC_AUTH_TOKEN}"
DATAMIND__LLM__PROTOCOL=anthropic
DATAMIND__LLM__MODEL=claude-sonnet-4-6
DATAMIND__LLM__FALLBACK_MODEL=claude-haiku-4-5
DATAMIND__LLM__TIMEOUT_S=90
DATAMIND__LLM__MAX_RETRIES=3

# ---- Embedding: DataMind's built-in huggingface provider -------------------
# BAAI/bge-m3 runs locally via sentence-transformers, so no embedding endpoint
# is needed (the gateway serves 82 chat models and zero embedding models).
# Multilingual on purpose: the same index serves the Chinese enterprise_demo
# profile and the English WorkSurface-Bench profiles.
#
# Weights live under $HF_HOME; hf-mirror is the reachable endpoint here
# (huggingface.co times out from this network).
HF_ENDPOINT=https://hf-mirror.com
# Weights are already cached locally, and hf-mirror is intermittently
# unreachable — without this, sentence-transformers spends ~25s on 5 retries
# revalidating config files at every startup before falling back to cache.
HF_HUB_OFFLINE=1
DATAMIND__EMBEDDING__PROVIDER=huggingface
DATAMIND__EMBEDDING__MODEL=BAAI/bge-m3
DATAMIND__EMBEDDING__DIMENSION=1024
DATAMIND__RETRIEVAL__STRATEGY=hybrid
DATAMIND__RETRIEVAL__TOP_K=6

# ---- Profile + budgets ----------------------------------------------------
DATAMIND__DATA__PROFILE=enterprise_demo
DATAMIND__AGENT__BACKEND=native
DATAMIND__AGENT__MAX_TURNS=14
DATAMIND__AGENT__MAX_TOOL_CALLS=28
DATAMIND__AGENT__WALL_CLOCK_TIMEOUT_S=300
DATAMIND__LOGGING__LEVEL=INFO

# sitecustomize.py here registers the `local_hash` provider at interpreter
# startup, so `python -m datamind` and `python -m benchmark.run` both see it.
PYTHONPATH="/vepfs-mlp2/c20250602/500050/lh/lx/2691/traj${PYTHONPATH:+:$PYTHONPATH}"

set +a

echo "[env] profile=${DATAMIND__DATA__PROFILE} model=${DATAMIND__LLM__MODEL}" \
     "embedding=${DATAMIND__EMBEDDING__PROVIDER} retrieval=${DATAMIND__RETRIEVAL__STRATEGY}"
