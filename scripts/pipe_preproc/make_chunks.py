from src.config import get_config
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.preprocess.md_filter import filt_and_save
from src.prompts.chunking_prompt import agentic_prompt, meta_prompt, decision_prompt
from src.preprocess.chunking import chunking_and_save

########### env load #######################################
CFG = get_config("configs/config_preproc.yaml")

MD_DIRS = [Path(d) for d in CFG.md_dirs]
CHUNK_DIR = Path(CFG.chunk_dir)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
'''
llm = ChatOpenAI(
    model=CFG.model_name,
    base_url="http://localhost:8000/v1",  # vLLM 또는 ollama 등 로컬 서버 주소
    api_key="EMPTY"  # 로컬 서버는 키 불필요
)
'''
meta_chain = meta_prompt | llm
chunking_chain = agentic_prompt | llm
decision_chain = decision_prompt | llm

##########################################################

if __name__ == "__main__":

    # filt_and_save(
    #     search_dirs=CFG.search_dirs,
    #     output_dir=CFG.output_dir,
    #     headers=CFG.headers,
    # )

    all_chunks = chunking_and_save(
        chain=chunking_chain,
        meta_chain=meta_chain,
        decision_chain=decision_chain,
        md_dirs=CFG.md_dirs,
        output_dir=CFG.chunk_dir,
        max_files=CFG.max_fnum
    )   