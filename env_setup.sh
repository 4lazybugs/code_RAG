# ########################## 콘다 설치 ######################################################
# # 콘다 설치 스크립트
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# # 콘다 설치 스크립트
# bash Miniconda3-latest-Linux-x86_64.sh
# # 콘다 초기화
# ~/miniconda3/bin/conda init
# # Anaconda 저장소(pkgs/main, pkgs/r)의 이용약관(ToS) 승인
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

########################## 가상환경 설치 ######################################################
# rag_env 가상환경 생성
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements_ragenv.txt
# paddle_env 가상환경 생성
conda create -n paddle_env python=3.10 -y
conda activate paddle_env
conda install pathlib -y
pip install -r requirements_paddleenv.txt


# ########################## vLLM 실행 ######################################################
# # 슈퍼컴(or VScode) 재부팅시 실행
# python -m vllm.entrypoints.openai.api_server \
#   --host 127.0.0.1 \
#   --port 8000 \
#   --model Qwen/Qwen2.5-14B-Instruct \
#   --max-model-len 32768 \
#   --gpu-memory-utilization 0.4