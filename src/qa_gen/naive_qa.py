def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def naive_qa(dir_path: Path, params: Params):
    files: List[Path] = [p for p in dir_path.glob("**/*.md") if p.is_file()]
    files.sort(key=lambda p: p.as_posix())

    result = []
    for p in tqdm(files, desc="Generating QA pairs"):
        md = read_text(p)

        question = f"{p.stem}에 대해 질문하라."
        answer = f"{p.stem}에 대한 답변을 작성하라."

        result.append({
            "id": p.stem,
            "question": question,
            "answer": answer,
        })

    return result