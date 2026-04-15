conda create -n docling_env python=3.10 -y
conda activate docling_env

python -m pip install --upgrade pip setuptools wheel
python -m pip install docling
python -m pip install -U pymupdf