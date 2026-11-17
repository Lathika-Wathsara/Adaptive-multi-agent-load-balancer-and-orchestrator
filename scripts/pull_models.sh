source "$(dirname "$0")/env.sh"
for m in qwen3:4b qwen3:14b deepseek-r1:7b qwen2.5vl:7b gemma3:4b nomic-embed-text; do
  run_in_sif ollama pull $m
done