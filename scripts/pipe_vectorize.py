import argparse, os, yaml, re
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

from src.preprocess.embedding import Embeddor, md2vecdb

########## config utils #######################################
def load_yaml(path):
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # 환경 변수 치환 처리
    config = {}
    for k, v in raw_config.items():
        if isinstance(v, str):
            config[k] = os.path.expandvars(v)
        else:
            config[k] = v

    return config

def get_config(path='configs/config_vec.yaml'):
    default_cfg = load_yaml(path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_cfg.get('model_name'))
    parser.add_argument("--metrics", type=str, default=",".join(default_cfg.get("metrics", [])))

    args, _ = parser.parse_known_args()

    # ✅ YAML + argparse 병합 → Namespace
    cfg = {**default_cfg, **vars(args)}
    return SimpleNamespace(**cfg)
#######################################################################

# 모듈 테스트 코드
if __name__ == "__main__":
    VEC_DIR = Path("db/vector_db")
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR = Path("db/raw_db_extracted/")

    CFG = get_config(path='configs/config_vec.yaml')
    EMBED_MODEL = CFG.embedor_model_name

    emb = Embeddor(EMBED_MODEL)

    md_files = list(EXTRACT_DIR.rglob("*.md"))

    groups = defaultdict(list)
    for p in md_files:
        rel = p.relative_to(EXTRACT_DIR)
        pdf_name = rel.parts[0]             # 1단계 폴더 기준
        groups[pdf_name].append(p)
        '''
        raw_db_extracted/   ← EXTRACT_DIR
        농장A/            ← pdf_name = rel.parts[0]
            report_1.md
            report_2.md
        농장B/            ← pdf_name = rel.parts[0]
            report_1.md
        
        ==> {
            "농장A"[pdf_name]: [Path("농장A/report_1.md"), Path("농장A/report_2.md")],
            "농장B"[pdf_name]: [Path("농장B/report_1.md")],
            }
        '''

    for pdf_name, _ in sorted(groups.items()):
        md_root = EXTRACT_DIR / pdf_name

        # pdf 이름에서 영문/숫자/_/-만 남기고 나머지는 _로 치환하고, 앞뒤 특수문자 제거하고, 63자로 자르는 코드
        pdf_name = f"cleaned_{re.sub(r'[^a-zA-Z0-9_-]+','_', pdf_name).strip('_')}"[:63].strip("_-.")
        
        md2vecdb(md_root=md_root, pdf_name=pdf_name, vector_dir=VEC_DIR, emb=emb)