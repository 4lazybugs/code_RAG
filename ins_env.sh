#!/bin/bash

# 1) 기존 환경 삭제 (있다면)
conda env remove -n rag_env -y

# 2) 환경 생성
conda create -n rag_env python=3.10 -y

# 3) 쉘에 Conda hook 로드
eval "$(conda shell.bash hook)"

# 4) 환경 활성화
conda activate rag_env

# 5) pip 최신화
pip install --upgrade pip

# 6) vllm 호환 버전으로 PyTorch 설치
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7) vllm 호환 버전들 설치
pip install transformers==4.53.2
pip install tiktoken==0.6.0
pip install openai==1.90.0
pip install tokenizers==0.21.1

# 8) vllm 설치
pip install vllm

# 9) 나머지 패키지들 설치 (충돌 없는 버전들)
pip install langchain==0.3.26
pip install langchain-core==0.3.68
pip install langchain-community==0.3.26
pip install langchain-ollama==0.3.4
pip install langchain-chroma>=0.1.0
pip install bitsandbytes==0.46.1
pip install accelerate==0.25.0
pip install sentencepiece>=0.1.99
pip install pandas>=2.0.0
pip install scipy>=1.11.0
pip install numpy>=1.26.2
pip install sqlalchemy
pip install pyarrow
pip install openpyxl
pip install pathlib
pip install PyPDF2
pip install PyMuPDF
pip install bert-score
pip install rouge_score==0.1.2
pip install evaluate==0.4.1
pip install sacrebleu==2.4.0
pip install tensorflow
pip install tf-keras
pip install tensorboard==2.14.0
pip install sentence-transformers
pip install nltk
pip install spacy
pip install datasets==2.15.0
pip install jsonlines
pip install filelock
pip install einops==0.7.0
pip install deepspeed==0.12.6
pip install peft==0.7.1
pip install packaging
pip install tqdm
pip install dataclasses
pip install metrics
pip install moverscore==1.0.3
pip install pot==0.9.3
pip install pydantic==2.11.7

echo "설치 완료! conda activate rag_env로 환경을 활성화하세요." 