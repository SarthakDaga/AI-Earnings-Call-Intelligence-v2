"""Turn-aware sentence extraction so every scored sentence knows its section."""

from src.ingestion.turns import _BOILER_PARA_RE, ParsedTranscript
from src.preprocessing.tokenizer import tokenize_sentences


def sentences_with_sections(parsed: ParsedTranscript) -> tuple[list[str], list[str]]:
    sentences, sections = [], []
    for t in parsed.turns:
        if t.role == "operator":
            continue
        for para in t.paragraphs:
            if _BOILER_PARA_RE.search(para):
                continue
            for s in tokenize_sentences(para):
                sentences.append(s)
                sections.append(t.section)
    return sentences, sections
