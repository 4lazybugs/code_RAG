########################## vLLM 실행 ######################################################

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen2.5-32B-Instruct \
  --max-model-len 6144 \
  --gpu-memory-utilization 0.9 \
  --swap-space 16 


# Qwen/Qwen2.5-3B-Instruct
# Qwen/Qwen2.5-14B-Instruct

#Qwen/Qwen2.5-32B-Instruct
# gpu - 0.95
# huggingface-cli login
# exaone은 그냥 script 다이렉트로 실행하면 됨 로컬에 모델 올릴필요없음