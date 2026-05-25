import re
from pathlib import Path
from types import SimpleNamespace

from src.preprocess.embedding import Embeddor, md2db, json2db, build_lexdb, build_chromadb
from src.config import get_config, load_yaml

# 모듈 테스트 코드
if __name__ == "__main__":
    VEC_DIR = Path("db/vector_db")
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR = Path("db/raw_db_extracted/")

    CFG_emb = get_config(path='configs/config_preproc.yaml')
    EMBED_MODEL = CFG_emb.embedor_model_name
    emb = Embeddor(EMBED_MODEL)

    CFG_fdir = get_config(path="configs/config_fdir.yaml")
    CHUNK_DIR = Path(CFG_fdir.chunk_dir)
    for chunk_root in CHUNK_DIR.iterdir():
        if not chunk_root.is_dir():
            continue

        pdf_name = chunk_root.name
        pdf_name = f"cleaned_{re.sub(r'[^a-zA-Z0-9_-]+','_', pdf_name).strip('_')}"[:63].strip("_-.")

        json2db(
            json_root=chunk_root,
            pdf_name=pdf_name,
            vector_dir=VEC_DIR,
            emb=emb
        )

    # for md_root in EXTRACT_DIR.iterdir():
    #     if not md_root.is_dir():
    #         continue
    #     '''
    #     raw_db_extracted/   ← EXTRACT_DIR
    #     농장A/            ← pdf_name
    #         report_1.md
    #         report_2.md
    #     농장B/            ← pdf_name
    #         report_1.md
    #     '''

    #     pdf_name = md_root.name
    #     # pdf 이름에서 영문/숫자/_/-만 남기고 나머지는 _로 치환하고, 앞뒤 특수문자 제거하고, 63자로 자르는 코드
    #     pdf_name = f"cleaned_{re.sub(r'[^a-zA-Z0-9_-]+','_', pdf_name).strip('_')}"[:63].strip("_-.")
        
    #     # chroma, lex 둘다
    #     md2db(md_root=md_root, pdf_name=pdf_name, vector_dir=VEC_DIR, emb=emb)
    #     # lex만
    #     md2db(md_root=md_root, pdf_name=pdf_name, vector_dir=VEC_DIR
    #         ,fns=[build_lexdb])
    #     # chroma만
    #     md2db(md_root=md_root, pdf_name=pdf_name, vector_dir=VEC_DIR, emb=emb,
    #         fns=[build_chromadb])