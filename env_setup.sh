# 콘다 설치 스크립트
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 콘다 설치 스크립트
bash Miniconda3-latest-Linux-x86_64.sh
# 콘다 초기화
~/miniconda3/bin/conda init
# Anaconda 저장소(pkgs/main, pkgs/r)의 이용약관(ToS) 승인
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 가상환경 설치
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements_ragenv.txt

# 의존성 해결기 + 설치기
pip install -U pip setuptools wheel  

# 슈퍼컴(or VScode) 재부팅시 실행
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen2.5-14B-Instruct \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.80



# The following command installs the PaddlePaddle version for CUDA 12.6. For other CUDA versions and the CPU version, please refer to https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
# For Linux systems, run:
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

pip install -r requirements.txt
# torch 2.6 + CUDA 12.4 설치:
conda install -c conda-forge openpyxl -y

chmod +x src/eval_pipe.sh