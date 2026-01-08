########################## vLLM 실행 ######################################################
# 터미널 초기화시 실행
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen2.5-32B-Instruct \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9