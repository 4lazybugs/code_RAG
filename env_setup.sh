conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install -r requirements.txt
# torch 2.6 + CUDA 12.4 설치:
conda install -c conda-forge openpyxl -y