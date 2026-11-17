source "$(dirname "$0")/env.sh"
nohup apptainer exec --nv --bind "$DATA":/data \
  --env OLLAMA_MODELS=/data/ollama_models \
  --env OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
  --env OLLAMA_KEEP_ALIVE=30m --env OLLAMA_MAX_LOADED_MODELS=3 \
  "$SIF" ollama serve > "$DATA/logs/ollama.log" 2>&1 &
sleep 5
curl -s http://127.0.0.1:$OLLAMA_PORT/api/tags && echo "ollama is up"