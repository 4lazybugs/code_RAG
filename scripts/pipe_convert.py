from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.config import get_config
from src.prompts.chunking_prompt import md2text_prompt, decision_prompt
from src.preprocess.parsing import md2text


#########  파라미터  #######################
CFG = get_config("configs/config_gen.yaml")

MD_DIRS   = [Path(d) for d in CFG.md_in_dirs]
CHUNK_DIR = Path(CFG.chunk_out_dir)

load_dotenv()

llm            = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
prose_chain    = md2text_prompt  | llm
decision_chain = decision_prompt | llm
###############################################


if __name__ == "__main__":
    load_dotenv()

    # ── 자연어 변환만 별도 폴더에 저장 (입력과 동일한 폴더 구조) ──
    md2text(
        md_dir=MD_DIRS,
        output_dir=CHUNK_DIR / "converted",
        prose_chain=prose_chain,
        decision_chain=decision_chain,
        use_decision_filter=True,
    )