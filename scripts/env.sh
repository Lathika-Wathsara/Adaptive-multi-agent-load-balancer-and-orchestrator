export CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATA=$HOME/data/Adaptive-multi-agent-load-balancer-and-orchestrator
export SIF=$DATA/paper_review.sif
export OLLAMA_PORT=11500

export APPTAINER_CACHEDIR=$HOME/data/.apptainer/cache
export APPTAINER_TMPDIR=$HOME/data/.apptainer/tmp

run_in_sif () {
  apptainer exec --nv \
    --bind "$CODE":/workspace --bind "$DATA":/data \
    --env OLLAMA_MODELS=/data/ollama_models \
    --env OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
    --pwd /workspace "$SIF" "$@"
}