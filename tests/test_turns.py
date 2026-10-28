from pathlib import Path

from src.ingestion.turns import (
    collapse_company_suffix,
    normalize_numbers,
    parse_transcript,
    to_clean_text,
    validate_header,
)
from src.preprocessing.chunker import chunk_turns, context_header

FIX = Path(__file__).parent / "fixtures" / "transcripts"


def _parse(name, **kw):
    return parse_transcript((FIX / name).read_text(), **kw)


def test_normalize_numbers():
    assert normalize_numbers("$68,000,000,000") == "$68 billion"
    assert normalize_numbers("$26,044,000,000") == "$26.044 billion"
    assert normalize_numbers("$1,350,000,000") == "$1.35 billion"
    assert normalize_numbers("$4,500,000") == "$4.5 million"
    assert normalize_numbers("$950,000 and $12.5") == "$950,000 and $12.5"


def test_collapse_company_suffix():
    assert collapse_company_suffix("NVIDIA Corporation reported") == "NVIDIA reported"
    assert collapse_company_suffix("Acme Corporation") == "Acme Corporation"


def test_turns_roles_and_sections():
    p = _parse("ACME_Q1_2025.txt")
    assert p.qa_found and p.analysts == ["Marcus Lee", "Elena Park"]
    by = {t.speaker: t for t in p.turns}
    assert by["Operator"].role == "operator"
    assert by["Rita Voss"].role == "ir"
    assert by["Daniel Okafor"].role == "executive"
    assert by["Marcus Lee"].role == "analyst" and by["Marcus Lee"].section == "qa"
    assert p.turns[2].section == "prepared"


def test_data_line_is_not_a_speaker():
    # F5: 'Revenue: $4.2B' must stay as text, not become a speaker label
    p = _parse("ACME_Q1_2025.txt")
    assert "Revenue" not in {t.speaker for t in p.turns}
    assert "Revenue: $4.2B" in to_clean_text(p)


def test_clean_text_drops_boilerplate_whole_paragraphs():
    p = _parse("ACME_Q1_2025.txt")
    text = to_clean_text(p)
    assert "forward-looking" not in text
    assert "listen-only" not in text
    assert "we may make  These" not in text  # v1 mid-sentence damage (F14)
    assert "$4.2 billion" in text            # F13 normalized


def test_missing_question_warning():
    p = _parse("ACME_Q2_2025.txt")
    assert any("Elena Park" in w for w in p.warnings)


def test_header_validation_flags_mismatch():
    p = _parse("ACME_Q1_2025.txt")
    problems = validate_header(p.header, fiscal_quarter="Q1", fiscal_year=2025, call_date="2025-04-30")
    assert any("Q2 FY2025" in x for x in problems)
    assert validate_header(p.header, fiscal_quarter="Q2", fiscal_year=2025, call_date="2025-04-30") == []
    assert any("date" in x for x in validate_header(p.header, fiscal_quarter="Q2", fiscal_year=2025,
                                                    call_date="2025-05-01"))


def test_chunks_keep_speaker_and_never_cross_sections():
    p = _parse("ACME_Q1_2025.txt")
    chunks = chunk_turns(p, 60, 10)
    assert chunks and all(c.speaker and c.role != "operator" for c in chunks)
    assert {c.section for c in chunks} == {"prepared", "qa"}
    assert "Q&A" in context_header("ACME", "Q1", 2025, [c for c in chunks if c.section == "qa"][0])


def test_chunk_overlap_and_size():
    from src.ingestion.turns import ParsedTranscript, Turn
    words = " ".join(f"w{i}" for i in range(500))
    p = ParsedTranscript("", [Turn(0, "A B", "executive", "prepared", text=words)])
    chunks = chunk_turns(p, 180, 40)
    assert all(len(c.text.split()) <= 180 for c in chunks)
    assert set(chunks[0].text.split()[-40:]) & set(chunks[1].text.split()[:40])


# ── label formats seen in real transcripts ───────────────────
LONG = "Revenue grew strongly this quarter across all of our major segments and regions worldwide. "


def test_bare_operator_and_dash_titles():
    raw = ("Hdr Corp\nQ1 Fiscal Year 2025 Earnings Call\n\n"
           "Operator\n\nGood afternoon. Welcome to the call.\n\n"
           "Jane Roe -- Vice President, Investor Relations\n\nWelcome everyone. " + LONG + "\n\n"
           "John Doe -- Chief Executive Officer\n\n" + LONG * 2 + "\n\n"
           "Ann Lee -- Chief Financial Officer\n\n" + LONG * 2 + "\n\n"
           "Operator\n\nThe first question comes from Sam Smith with Big Bank. Your line is open.\n\n"
           "Sam Smith -- Analyst, Big Bank\n\nHow is demand holding up this quarter and next year please?\n\n"
           "John Doe -- Chief Executive Officer\n\nDemand is strong. " + LONG)
    p = parse_transcript(raw)
    assert p.qa_found and p.analysts == ["Sam Smith"]
    roles = [t.role for t in p.turns]
    assert roles.count("operator") == 2 and "analyst" in roles
    assert p.turns[0].speaker == "Operator" and "Good afternoon" in p.turns[0].text
    assert [t.section for t in p.turns][-1] == "qa" and p.turns[1].section == "prepared"


def test_named_operator_with_colon():
    raw = ("Hdr\n\nJane Roe: Welcome everyone. " + LONG + "\nJohn Doe: " + LONG * 2 +
           "\nAnn Lee: " + LONG * 2 + "\nSarah: The next question comes from Sam Smith with Big Bank. Your line is open.\n"
           "Sam Smith: How is demand holding up this quarter and next year please?\n"
           "John Doe: Demand is strong. " + LONG + "\n"
           "Sarah: Your next question comes from Pat Jones with Other Bank. Your line is open.\n"
           "Pat Jones: What about margins over the coming year given current costs?\n")
    p = parse_transcript(raw)
    assert p.qa_found and len(p.analysts) == 2
    assert sum(t.role == "operator" for t in p.turns) == 2


def test_bare_names_and_comma_titles():
    raw = ("Hdr\n\nTim Cook, Chief Executive Officer\n\n" + LONG * 2 + "\n\n"
           "Luca Maestri, Chief Financial Officer\n\n" + LONG * 2 + "\n\n"
           "Operator\n\nThe first question comes from Sam Smith with Big Bank.\n\n"
           "Sam Smith\n\nHow is demand holding up this quarter and next year please?\n\n"
           "Tim Cook\n\nDemand is strong. " + LONG + "\n\n"
           "Operator\n\nThe next question comes from Pat Jones with Other Bank.\n\n"
           "Pat Jones\n\nWhat about margins over the coming year given current costs?\n")
    p = parse_transcript(raw)
    assert [t.role for t in p.turns].count("analyst") == 2
    assert p.qa_found and p.turns[0].speaker == "Tim Cook" and p.turns[0].role == "executive"


def test_headings_are_not_speakers():
    raw = ("Hdr\n\nJane Roe: Welcome. " + LONG + "\n\nPrepared Remarks\n\n" + LONG +
           "\n\nQuestions and Answers\n\n" + LONG)
    p = parse_transcript(raw)
    assert {t.speaker for t in p.turns} <= {"Jane Roe", "Unknown"}
    assert not any("Questions and Answers" in t.text or "Prepared Remarks" in t.text for t in p.turns)


def test_initials_in_analyst_name():
    raw = ("Hdr\n\nJane Roe: Welcome. " + LONG + "\nJohn Doe: " + LONG + "\nAnn Lee: " + LONG +
           "\nOperator: The next question comes from C.J. Muse with Cantor Fitzgerald.\n"
           "C.J. Muse: How is demand holding up this quarter and next year please?\n")
    p = parse_transcript(raw)
    assert p.analysts == ["C.J. Muse"] and p.turns[-1].role == "analyst"


def test_participant_list_with_hyphens_and_bare_names():
    """Seeking-Alpha style: 'Name - Title' list at the top, then names alone on a line."""
    raw = ("Corp Q1 2026 Earnings Call January 29, 2026\n\nCompany Participants\n\n"
           "Pam Roe - Director of Investor Relations\nTim Doe - CEO & Director\nAnn Lee - Senior VP & CFO\n\n"
           "Conference Call Participants\n\nSam Smith - Big Bank Research Division\nPat Jones - Other Bank\n\n"
           "Operator\n\nGood day, everyone. Welcome to the call.\n\n"
           "Pam Roe\n\nThank you. Good afternoon and welcome. " + LONG + "\n\n"
           "Tim Doe\n\n" + LONG * 2 + "\n\nAnn Lee\n\n" + LONG * 2 + "\n\n"
           "Pam Roe\n\nWe will now open for questions. Operator, please go ahead.\n\n"
           "Operator\n\nThank you. Our first question comes from Sam Smith with Big Bank.\n\n"
           "Sam Smith\n\nHow is demand holding up this quarter and next year please?\n\n"
           "Tim Doe\n\nDemand is strong. " + LONG + "\n\n"
           "Sam Smith\n\nThanks.\n\n"
           "Operator\n\nOur next question comes from Pat Jones with Other Bank.\n\n"
           "Pat Jones\n\nWhat about margins over the coming year given costs?\n")
    p = parse_transcript(raw)
    by = {t.speaker: t.role for t in p.turns}
    assert by["Pam Roe"] == "ir" and by["Tim Doe"] == "executive" and by["Sam Smith"] == "analyst"
    assert by["Operator"] == "operator"           # the IR person must NOT be mistaken for the operator
    assert p.qa_found and p.turns[0].speaker == "Operator" and "Pam Roe" in {t.speaker for t in p.turns}
    assert not any(t.speaker == "Pam Roe" and t.section == "qa" and t.role == "operator" for t in p.turns)
    assert not p.warnings                          # "Thanks." (a later analyst turn) is not a truncation


def test_analyst_label_with_firm_and_no_operator():
    """MSFT style: 'NAME, Firm:' labels, no operator turns, IR announces Q&A."""
    raw = ("Hdr\n\nJON NEIL:\nGood afternoon. " + LONG + "\n\nSATYA NAD:\n" + LONG * 2 + "\n\nAMY HOOD:\n" + LONG * 2 +
           "\n\nJON NEIL:\nThanks, Amy. We will now move over to Q&A.\n\n(Operator Direction.)\n\n"
           "KEITH WEISS, Morgan Stanley: How is demand holding up this quarter and next year please?\n\n"
           "SATYA NAD:\nDemand is strong. " + LONG + "\n\nJON NEIL:\nThanks, Keith. Operator, next question, please.\n\n"
           "(Operator Direction.)\n\nBRENT THILL, Jefferies: What about margins over the coming year given costs?\n")
    p = parse_transcript(raw)
    assert p.qa_found and len(p.analysts) == 2
    assert not any(t.role == "operator" for t in p.turns)
    assert not any("Operator Direction" in t.text for t in p.turns)
    first_qa = next(t for t in p.turns if t.section == "qa")
    assert first_qa.speaker == "JON NEIL"           # the IR line that announces Q&A opens the section


def test_missing_labels_keep_text_and_use_heading():
    """NVDA-Q2-2026 style: only 'Operator:' is labelled; everything else is unlabelled text."""
    body = (LONG * 12 + "\n\n") * 6
    raw = ("Hdr\n\nOperator: Good afternoon. My name is Sarah and I am your conference operator.\n\n" + body +
           "QUESTION AND ANSWER SECTION\n\nOperator: Thank you. Your first question comes from Sam Smith with Big Bank. "
           "Your line is open.\n\n" + body)
    p = parse_transcript(raw)
    assert p.qa_found
    assert not any(t.role == "operator" and t.word_count > 200 for t in p.turns)  # never drop real speech
    assert any("no speaker label" in w for w in p.warnings)
    assert "Revenue grew strongly" in to_clean_text(p)


def test_operator_only_labels_split_operator_from_speech():
    """GOOGL style: 'Operator:' is labelled; speakers are not. Operator lines must not swallow the speech."""
    sp = (LONG * 8 + "\n\n") * 3
    raw = ("Hdr\n\nOperator: Good afternoon. My name is Sarah and I am your conference operator.\n\n"
           "Toshiya Hari, you may begin your conference.\n\n" + sp +
           "Operator: Thank you. Your first question comes from Sam Smith with Big Bank. Your line is open.\n\n" + sp +
           "Operator: Your next question comes from Pat Jones with Other Bank. Your line is open.\n\n" + sp +
           "Operator: This concludes the question-and-answer session.\n")
    p = parse_transcript(raw)
    ops = [t for t in p.turns if t.role == "operator"]
    assert len(ops) == 4 and all(t.word_count < 60 for t in ops)       # only the operator's own words
    unk = [t for t in p.turns if t.role == "unknown"]
    assert len(unk) == 3 and all(t.word_count > 150 for t in unk)      # speech preserved, speaker not credited
    assert p.qa_found and p.analysts == ["Sam Smith", "Pat Jones"]
    clean = to_clean_text(p)
    assert "Your line is open" not in clean and "Revenue grew strongly" in clean