#!/usr/bin/env bash

##### 실행 방법 ################
#chmod +x setup.sh #######
##### ./setup.sh #############

set -e

# 1) 시스템 의존성 설치
sudo apt update
sudo apt install -y git git-lfs

# 2) Conda 패키지 설치
conda install -y pandas
conda install -y -c conda-forge fastparquet

# 3) Ollama 설치 (공식 설치 방법)
# Remove snap version if exists
sudo snap remove ollama 2>/dev/null || true

# Install Ollama using official method
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 5  # Wait for service to start

# embedding model (data -> vector)
ollama pull mxbai-embed-large
ollama pull sqlcoder:15b
ollama pull mistral-nemo
# ollama model
ollama pull llama3.2
