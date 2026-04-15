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
#############################
##### rag_env 가상환경 생성 ####
#############################
conda create -n rag_env python=3.10 -y
conda activate rag_env
pip install git+https://github.com/google-research/bleurt.git # bleurt 설치 <- evaluator 중 하나
#pip install git+https://github.com/Liyan06/MiniCheck.git # mini_chek 설치 <- judge_lm
pip install -r requirements.txt # 각종 rag_env의 requirement 설치
python -c "import nltk; nltk.download('punkt_tab')" # NLTK 리소스 다운로드