from typing import List


def tokenize_kiwi(text: str) -> List[str]:
    from kiwipiepy import Kiwi
    if not hasattr(tokenize_kiwi, "_kiwi"):
        tokenize_kiwi._kiwi = Kiwi()
    return [t.form for t in tokenize_kiwi._kiwi.tokenize(text)]


def tokenize_moses(text: str) -> List[str]:
    from sacremoses import MosesTokenizer
    if not hasattr(tokenize_moses, "_moses"):
        tokenize_moses._moses = MosesTokenizer()
    return tokenize_moses._moses.tokenize(text, return_str=False)


def tokenize_spacy(text: str) -> List[str]:
    import spacy
    if not hasattr(tokenize_spacy, "_nlp"):
        tokenize_spacy._nlp = spacy.load("en_core_web_sm")
    doc = tokenize_spacy._nlp.make_doc(text)
    return [t.text for t in doc if not t.is_space]


TOKENIZER_MAP = {
    "kiwi": tokenize_kiwi,
    "moses": tokenize_moses,
    "spacy": tokenize_spacy,
}


def get_tokenizer(name: str):
    if name not in TOKENIZER_MAP:
        raise ValueError(f"Unknown tokenizer: {name}")
    return TOKENIZER_MAP[name]
