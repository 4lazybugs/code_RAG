from pathlib import Path
from src.preprocess.md_filter import trim_and_save

if __name__ == "__main__":
    SEARCH_DIRS = [
        Path("db/raw_db_extracted/consulting_2022_01"),
        Path("db/raw_db_extracted/consulting_2022_02"),
        Path("db/raw_db_extracted/consulting_2022_04"),
        Path("db/raw_db_extracted/consulting_2024_(1)"),
        Path("db/raw_db_extracted/consulting_2024_(2)"),
    ]

    OUTPUT_DIR = Path("db/qa_synthesis_md/")

    HEADERS = [
        "# 1차 현장컨설팅 결과보고서",
        "# 1 차 현장컨설팅 결과보고서",
        "# 2차 현장컨설팅 결과보고서",
        "# 2 차 현장컨설팅 결과보고서",
        '<div style="text-align: center;">1차 현장 컨설팅 결과보고서</div>',
        '<div style="text-align: center;">1차 현장컨설팅 결과보고서</div>',
        '<div style="text-align: center;">2차 현장 컨설팅 결과보고서</div>',
        '<div style="text-align: center;">2차 현장컨설팅 결과보고서</div>',
    ]

    trim_and_save(
        search_dirs=SEARCH_DIRS,
        output_dir=OUTPUT_DIR,
        headers=HEADERS,
    )