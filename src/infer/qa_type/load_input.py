from typing import Dict, Any, Optional, Sequence

def mcq_input(raw_input: Dict[str, Any]) -> Dict[str, str]:
    
    #options 리스트를 프롬프트에 넣을 문자열로 변환하는 함수
    def _format_options(options: Optional[Sequence[str]]) -> str:
        if not options:
            return ""
        return "[선택지]\n" + "\n".join(options)

    return {
        "id": raw_input["id"],
        "question": raw_input["question"],
        "answer": raw_input["answer"],
        "options": _format_options(raw_input.get("options")),
    }

def saq_input(raw_input: Dict[str, Any]) -> Dict[str, str]:

    return {
        "id": raw_input["id"],
        "question": raw_input["question"],
        "answer": raw_input["answer"],
    }
