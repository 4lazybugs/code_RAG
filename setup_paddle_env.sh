#!/usr/bin/env bash
set -e

conda create -n paddle_env python=3.10 -y
conda install -n paddle_env -c conda-forge libstdcxx-ng -y

conda run -n paddle_env python -m pip install --upgrade pip setuptools wheel
conda run -n paddle_env python -m pip install requests typing_extensions jinja2 beautifulsoup4
conda run -n paddle_env python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
conda run -n paddle_env python -m pip install "paddleocr[all]"
conda run -n paddle_env python -m pip install -U pymupdf