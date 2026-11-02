"""
src/ingestion/turns.py

Structure-first transcript parser (R0.4, R0.7, R0.9).

v1 stripped speaker labels and removed boilerplate mid-sentence, which left
fragments like "we may make  These are subject to...". This parser instead:

  1. splits the call into speaker TURNS (paragraphs without a label continue
     the previous speaker),
  2. finds the prepared-remarks / Q&A boundary,
  3. infers a coarse speaker ROLE (operator / ir / executive / analyst),
  4. normalizes number formats ("$68,000,000,000" -> "$68 billion"),
  5. drops boilerplate by whole paragraph or whole turn, never mid-sentence,
  6. reports warnings (e.g. an analyst turn with no question body).

Roles are best-effort: source labels are noisy. Do not treat role filters as
exact.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation

# ── number normalization ─────────────────────────────────────
_MONEY_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+)(?!\d)(?!\.\d)")


def _scale(match: re.Match) -> str:
    try:
        value = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:  # pragma: no cover
        return match.group(0)
    for unit, size in (("trillion", Decimal(10) ** 12), ("billion", Decimal(10) ** 9),
                       ("million", Decimal(10) ** 6)):
        if value >= size:
            scaled = (value / size).normalize()
            text = format(scaled, "f")
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"${text} {unit}"
    return match.group(0)  # below 1 million: leave as is


def normalize_numbers(text: str) -> str:
    """'$68,000,000,000' -> '$68 billion'; '$26,044,000,000' -> '$26.044 billion'."""
    return _MONEY_RE.sub(_scale, text)


def collapse_company_suffix(text: str, names: tuple[str, ...] = ("NVIDIA", "Microsoft", "Apple",
                                                                  "Amazon", "Alphabet", "Google")) -> str:
    """'NVIDIA Corporation' -> 'NVIDIA' (F17; optional)."""
    pattern = r"\b(" + "|".join(map(re.escape, names)) + r")\s+(?:Corporation|Corp\.?|Inc\.?)(?=[\s,.;:)])"
    return re.sub(pattern, r"\1", text)


# ── data model ───────────────────────────────────────────────
@dataclass
class Turn:
    idx: int
    speaker: str
    role: str = "unknown"      # operator | ir | executive | analyst | unknown
    section: str = "prepared"  # prepared | qa
    title: str = ""
    text: str = ""
    qa_marker: bool = False   # a "QUESTION AND ANSWER SECTION" heading came right before this turn

    @property
    def paragraphs(self) -> list[str]:
        return [p for p in self.text.split("\n") if p.strip()]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ParsedTranscript:
    header: str
    turns: list[Turn]
    warnings: list[str] = field(default_factory=list)
    qa_found: bool = False
    analysts: list[str] = field(default_factory=list)


# ── speaker-label detection ──────────────────────────────────
# A name token: "Jensen", "O'Brien", "NADELLA", "J.", or initials like "C.J."
_NAME = r"(?:[A-Z][A-Za-z'’\-]+\.?|(?:[A-Z]\.){1,3})"
_COLON_RE = re.compile(rf"^(?P<name>{_NAME}(?: {_NAME}){{0,3}}):\s*(?P<rest>.*)$")
_DASH_RE = re.compile(rf"^(?P<name>{_NAME}(?: {_NAME}){{1,3}})\s+(?:--|—|–|-)\s+(?P<title>[A-Za-z ,&.\-/]{{3,90}})$")
# "KEITH WEISS, Morgan Stanley: question text" (analyst label with firm)
_ANALYST_COLON_RE = re.compile(
    rf"^(?P<name>{_NAME}(?: {_NAME}){{1,3}})\s*,\s*(?P<firm>[A-Za-z&.'’\- ]{{2,50}}):\s*(?P<rest>.*)$")
_QA_HEADING_RE = re.compile(r"^(?:questions?\s+(?:and|&)\s+answers?|q\s*&\s*a)(?:\s+(?:section|session))?\s*:?$", re.I)
_HEADING_LINE_RE = re.compile(r"^(?:prepared remarks|presentation|opening remarks|management discussion(?: section)?"
                              r"|introduction)\s*:?$", re.I)
_PLIST_ANALYST_RE = re.compile(r"^(?:conference call participants|call participants|analysts?(?: participants)?|"
                               r"analyst participants)\s*:?$", re.I)
_PLIST_EXEC_RE = re.compile(r"^(?:company participants|corporate participants|company representatives|"
                            r"executives?(?: participants)?)\s*:?$", re.I)
_STAGE_DIR_RE = re.compile(r"^\s*\((?:operator direction|laughter|applause|inaudible|crosstalk|pause)[^)]*\)\.?\s*$",
                           re.I | re.M)
# "Tim Cook, Chief Executive Officer" / "Tim Cook (CEO)" on its own line
_BARE_COMMA_RE = re.compile(rf"^(?P<name>{_NAME}(?: {_NAME}){{1,3}})\s*[,(]\s*(?P<title>[^\n]{{3,90}}?)\)?$")
_BARE_PLAIN_RE = re.compile(rf"^(?P<name>{_NAME}(?: {_NAME}){{1,3}})$")
_BARE_OP_RE = re.compile(r"^(operator|moderator|host)$", re.I)
_SINGLE_OK = {"Operator", "Moderator", "Host"}
# Words that make "Label: value" a data line / heading rather than a speaker
_NOT_NAME = {
    "revenue", "gaap", "non-gaap", "total", "note", "notes", "question", "questions", "answer", "answers",
    "operating", "net", "gross", "income", "margin", "guidance", "date", "ticker", "company", "title",
    "source", "fiscal", "year", "quarter", "call", "q&a", "segment", "data", "center",
    "earnings", "eps", "cash", "outlook", "summary", "highlights", "participants", "page",
    "slide", "free", "flow", "growth", "sales", "services", "products", "cloud", "gaming",
    "presentation", "prepared", "remarks", "introduction", "conclusion", "appendix", "disclaimer",
    "executives", "analysts", "overview", "transcript", "conference", "corporation", "inc", "corp",
    "operator", "and",
}
_TITLE_WORD_RE = re.compile(
    r"\b(CEO|CFO|COO|CTO|Chief|Officer|President|Chairman|Director|Analyst|Investor Relations|"
    r"Vice President|VP|SVP|EVP|Head|Managing|Partner|Research|Treasurer|Controller|Founder)\b", re.I)
# Text that only a conference operator says. Used to accept "Sarah:" as a label and to find the operator.
_OPERATOR_TEXT_RE = re.compile(
    r"(your line is open|questions? (?:comes|come|is coming) from|comes from the line of|conference operator|"
    r"operator instructions|placed on mute|you may now disconnect|"
    r"concludes (?:today's|the) (?:conference|question|q&a))", re.I)


def _is_title_line(s: str) -> bool:
    """A job-title line printed directly under a speaker name, e.g.
    'Managing Director and Senior Equity Research Analyst, Barclays' or 'VP of Investor Relations, Amazon'."""
    return (3 <= len(s) <= 120 and len(s.split()) <= 18 and not re.search(r"[.!?:]$", s)
            and bool(_TITLE_WORD_RE.search(s)) and not _BARE_OP_RE.match(s)
            and not _OP_LINE_RE.search(s) and not re.search(r"\bquestions?\b", s, re.I))  # operator speech is not a title


# A paragraph that is plausibly still the operator talking (used to split "Operator:" blocks)
_OP_LINE_RE = re.compile(_OPERATOR_TEXT_RE.pattern + r"|you may begin|please go ahead|^\s*thank you[.!]?\s*$"
                         r"|^\s*\[operator", re.I)


def _is_speaker_label(name: str, rest: str, known: set[str]) -> bool:
    if name in known:
        return True
    tokens = name.split()
    if len(tokens) == 1:
        # single word: only "Operator" or a named operator ("Sarah: The next question comes from...")
        return name in _SINGLE_OK or bool(_OPERATOR_TEXT_RE.search(rest))
    if any(t.lower().strip(".") in _NOT_NAME for t in tokens):
        return False
    if rest and re.match(r"^[\$\d%(]", rest):  # "Gross Profit: $5B"
        return False
    return True


def _name_ok(name: str) -> bool:
    return not any(t.lower().strip(".,") in _NOT_NAME for t in name.split())


def _next_nonblank(lines: list[str], i: int) -> str:
    for ln in lines[i + 1:]:
        if ln.strip():
            return ln.strip()
    return ""


def _classify(lines: list[str], i: int, known: set[str], repeats: Counter,
              intro_names: tuple[str, ...] = ()) -> tuple[str, str, str] | None:
    """Return (speaker, title, rest_of_line) if lines[i] starts a speaker turn, else None."""
    s = lines[i].strip()
    if not s:
        return None
    ac = _ANALYST_COLON_RE.match(s)
    if ac and _name_ok(ac["name"]):
        return ac["name"], "Analyst, " + ac["firm"].strip(), ac["rest"].strip()
    m = _COLON_RE.match(s)
    if m and _is_speaker_label(m["name"], m["rest"], known):
        return m["name"], "", m["rest"].strip()
    d = _DASH_RE.match(s)
    if d:
        return d["name"], d["title"].strip(), ""
    # Labels on their own line need a blank line before them (or start of file).
    if (i > 0 and lines[i - 1].strip()) or len(s) > 90:
        return None
    if _BARE_OP_RE.match(s):
        return s.title(), "", ""
    c = _BARE_COMMA_RE.match(s)
    if c and _name_ok(c["name"]) and _TITLE_WORD_RE.search(c["title"]):
        return c["name"], c["title"].strip(), ""
    p = _BARE_PLAIN_RE.match(s)
    if p and _name_ok(p["name"]):
        name = p["name"]
        if name in known or repeats[name] >= 2 or any(_same_person(name, n) for n in intro_names):
            return name, "", ""
        if i + 1 < len(lines) and _is_title_line(lines[i + 1].strip()):  # "Name" + "Title, Firm" + speech
            return name, "", ""
        if len(_next_nonblank(lines, i).split()) >= 12:  # a one-off name followed by a real paragraph
            return name, "", ""
    return None


def _same_person(a: str, b: str) -> bool:
    ta, tb = a.replace(".", "").split(), b.replace(".", "").split()
    return bool(ta and tb and ta[-1].lower() == tb[-1].lower() and ta[0][0].lower() == tb[0][0].lower())


# ── Q&A boundary + analyst intros ────────────────────────────
_QA_START_RE = re.compile(
    r"(first question|question (?:is|comes|will come) from|open (?:up )?the (?:call|floor|line)"
    r"|take (?:our |your )?(?:first )?question|q&a session|question-and-answer session"
    r"|(?:move|turn|go)(?: over| on)? to (?:the )?q&a|q&a (?:portion|section|part)"
    r"|we(?:'ll| will) now (?:begin|start|take|open))", re.I)
_ANALYST_INTRO_RE = re.compile(
    rf"question[s]?\s+(?:comes|is|will come|coming)\s+from\s+(?:the line of\s+)?"
    rf"(?P<name>{_NAME}(?: {_NAME}){{0,3}})\s+(?:with|of|from|at|representing)\b")
_BOILER_PARA_RE = re.compile(
    r"(forward[\-\s]looking statements?|safe harbor|this call is being recorded|"
    r"no part of this (?:call|transcript)|may not be reproduced|replay of (?:today's|this) call|"
    r"all participants are in (?:a )?listen[\-\s]only)", re.I)
_EXEC_TITLE_RE = re.compile(r"\b(CEO|CFO|COO|CTO|President|Chief|Officer|Chairman|EVP|SVP|Vice President|VP)\b", re.I)
_IR_TITLE_RE = re.compile(r"investor relations|\bIR\b", re.I)


def parse_transcript(raw_text: str, *, normalize: bool = True,
                     collapse_company: bool = False) -> ParsedTranscript:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\[\(]\d{2}:\d{2}(?::\d{2})?[\]\)]", "", text)  # timestamps
    text = _STAGE_DIR_RE.sub("", text)  # "(Operator Direction.)", "(Laughter)"
    lines = [ln.rstrip() for ln in text.split("\n")]

    # Names that appear alone on a line at least twice are speakers in a "name on its own line" format.
    repeats: Counter = Counter()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if (i == 0 or not lines[i - 1].strip()) and _BARE_PLAIN_RE.match(s) and _name_ok(s):
            repeats[s] += 1

    # Analysts the operator announces ("question comes from Sam Smith with ...") are speakers too.
    intro_names = tuple({m["name"] for m in _ANALYST_INTRO_RE.finditer(text)})

    turns: list[Turn] = []
    known: set[str] = set()
    cur: Turn | None = None
    header_lines: list[str] = []
    plist_analyst = False

    def start(name: str, title: str = "") -> Turn:
        t = Turn(idx=len(turns), speaker=name, title=title)
        turns.append(t)
        known.add(name)
        return t

    blank_seen = False
    title_lines: set[int] = set()  # title lines already attached to a speaker (never part of the speech)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            blank_seen = True
            continue
        if i in title_lines:
            continue
        after_blank, blank_seen = blank_seen, False
        no_text_yet = not any(t.text for t in turns)
        if no_text_yet and (_PLIST_ANALYST_RE.match(s) or _PLIST_EXEC_RE.match(s)):
            plist_analyst = bool(_PLIST_ANALYST_RE.match(s))  # participant lists: titles only, no text
            continue
        if _HEADING_LINE_RE.match(s):
            plist_analyst = False  # "Presentation": the participant lists are over
            continue  # section headings carry no speech
        if _QA_HEADING_RE.match(s):
            cur = start("Unknown")
            cur.qa_marker = True
            continue
        hit = _classify(lines, i, known, repeats, intro_names)
        if hit:
            name, title, rest = hit
            consumed = False
            if (not rest and not title and name not in _SINGLE_OK and i + 1 < len(lines)
                    and _is_title_line(lines[i + 1].strip())):
                title = lines[i + 1].strip()  # "Name\nTitle, Firm\nspeech": the title is not speech
                title_lines.add(i + 1)
                consumed = True
            if (no_text_yet and plist_analyst and not rest and title and not consumed
                    and not title.lower().startswith("analyst")):
                title = "Analyst, " + title
            cur = start(name, title)
            if rest:
                cur.text = rest
            continue
        if cur is None:
            header_lines.append(s)  # everything before the first speaker label
            continue
        if after_blank and cur.speaker in _SINGLE_OK and cur.text and not _OP_LINE_RE.search(s):
            # Some sources label only "Operator:"; the next paragraph after the operator's own is someone else's speech.
            cur = start("Unknown")
        cur.text = (cur.text + "\n" + s).strip() if cur.text else s

    header = "\n".join(header_lines)

    # Participant lists ("Tim Cook -- CEO" with no text) only carry titles: share them, then drop the empty turns.
    titles: dict[str, str] = {}
    for t in turns:
        if t.title:
            titles.setdefault(t.speaker, t.title)
    carry, kept = False, []
    for t in turns:  # a heading with no text of its own hands its marker to the next turn
        if t.qa_marker and not t.text.strip():
            carry = True
            continue
        if carry:
            t.qa_marker, carry = True, False
        kept.append(t)
    turns = kept
    first_text = next((i for i, t in enumerate(turns) if t.text.strip()), len(turns))
    turns = turns[first_text:]  # empty turns after this point are kept: they signal a truncated source (F16)
    for i, t in enumerate(turns):
        t.idx = i
        t.title = titles.get(t.speaker, t.title)  # first title wins: the participant list says who is an analyst

    # ── operator(s): literal "Operator", or whoever speaks only operator lines ("Sarah:") ──
    op_names = {t.speaker for t in turns if t.speaker in _SINGLE_OK}
    if not op_names:  # no literal "Operator": look for a named one ("Sarah:") who only says operator lines
        for name in {t.speaker for t in turns}:
            mine = [t for t in turns if t.speaker == name]
            hits = sum(1 for t in mine if _OPERATOR_TEXT_RE.search(t.text))
            avg_words = sum(t.word_count for t in mine) / len(mine)
            has_title = any(t.title for t in mine)
            if hits >= 2 and hits / len(mine) >= 0.6 and avg_words < 80 and not has_title:
                op_names.add(name)

    # ── analysts, Q&A boundary ──
    analysts: list[str] = []
    for t in turns:
        if t.speaker in op_names:
            for m in _ANALYST_INTRO_RE.finditer(t.text):
                if not any(_same_person(m["name"], a) for a in analysts):
                    analysts.append(m["name"])
        elif re.search(r"\banalyst\b", t.title, re.I) and not any(_same_person(t.speaker, a) for a in analysts):
            analysts.append(t.speaker)

    def is_analyst_name(n: str) -> bool:
        return any(_same_person(n, a) for a in analysts)

    qa_idx = next((t.idx for t in turns if t.qa_marker), None)
    if qa_idx is None:
        non_op_seen = 0
        for t in turns:
            if t.speaker in op_names:
                announces = bool(_ANALYST_INTRO_RE.search(t.text))  # "question comes from X with Firm"
                if _QA_START_RE.search(t.text) and (non_op_seen >= 3 or (non_op_seen >= 1 and announces)):
                    qa_idx = t.idx
                    break
            else:
                non_op_seen += 1
    if qa_idx is None:
        for t in turns:
            if t.speaker not in op_names and is_analyst_name(t.speaker) and t.idx > 0:
                qa_idx = t.idx
                for _ in range(2):  # include the operator / IR line that announces Q&A
                    prev = turns[qa_idx - 1] if qa_idx > 0 else None
                    if prev and (prev.speaker in op_names
                                 or (prev.word_count < 150 and _QA_START_RE.search(prev.text))):
                        qa_idx -= 1
                break
    for t in turns:
        t.section = "qa" if (qa_idx is not None and t.idx >= qa_idx) else "prepared"

    # ── roles ──
    prepared_speakers = {t.speaker for t in turns
                         if t.section == "prepared" and t.speaker not in op_names and t.speaker != "Unknown"}
    first_non_op = next((t for t in turns if t.speaker not in op_names and t.speaker != "Unknown"), None)
    for t in turns:
        if t.speaker in op_names:
            t.role = "operator"
        elif t.speaker == "Unknown":
            t.role = "unknown"  # unlabeled speech: kept, but nobody is credited with it
        elif is_analyst_name(t.speaker) or re.search(r"\banalyst\b", t.title, re.I):
            t.role = "analyst"
        elif _IR_TITLE_RE.search(t.title) or (t is first_non_op and re.search(r"\bwelcome\b", t.text[:300], re.I)
                                             and not _EXEC_TITLE_RE.search(t.title)):
            t.role = "ir"
        elif _EXEC_TITLE_RE.search(t.title) or t.speaker in prepared_speakers:
            t.role = "executive"
        elif t.section == "qa":
            t.role = "analyst" if "?" in t.text else "executive"
    # propagate ir/exec identity across a speaker's turns
    role_by_speaker: dict[str, str] = {}
    for t in turns:
        if t.role in ("ir", "executive", "analyst"):
            role_by_speaker.setdefault(t.speaker, t.role)
    for t in turns:
        if t.role == "unknown" and t.speaker in role_by_speaker:
            t.role = role_by_speaker[t.speaker]

    # A huge "operator" turn is speech whose speaker labels are missing in the source. Keep it (never drop it).
    unlabeled_words = sum(t.word_count for t in turns if t.speaker == "Unknown")
    for t in turns:
        if t.role == "operator" and t.word_count > 200:
            unlabeled_words += t.word_count
            t.role = "unknown"

    # ── normalization ──
    if normalize or collapse_company:
        for t in turns:
            if normalize:
                t.text = normalize_numbers(t.text)
            if collapse_company:
                t.text = collapse_company_suffix(t.text)

    # ── warnings ──
    warnings: list[str] = []
    total_words = sum(t.word_count for t in turns) or 1
    if unlabeled_words > 1500:
        warnings.append(f"source labels only some speakers: {unlabeled_words} words "
                        f"({unlabeled_words * 100 // total_words}%) have no speaker label (kept, role unknown)")
    seen_analysts: set[str] = set()
    for t in turns:
        first_turn = t.speaker not in seen_analysts
        if t.role == "analyst" and t.section == "qa":
            seen_analysts.add(t.speaker)
        # only an analyst's FIRST turn should hold a real question; later short turns are "Thanks."
        if t.role == "analyst" and t.section == "qa" and first_turn and t.word_count < 8:
            warnings.append(f"turn {t.idx}: analyst '{t.speaker}' has a very short or missing question "
                            f"({t.word_count} words) - possible truncated source")
    if qa_idx is None:
        warnings.append("no Q&A section detected")
    if not turns:
        warnings.append("no speaker turns detected")

    return ParsedTranscript(header=header, turns=turns, warnings=warnings,
                            qa_found=qa_idx is not None, analysts=analysts)


def to_clean_text(parsed: ParsedTranscript, include_roles: tuple[str, ...] | None = None) -> str:
    """
    Analysis-ready text for FinBERT/summaries: operator turns and boilerplate
    paragraphs (whole paragraphs only) are dropped; speaker labels removed.
    """
    out: list[str] = []
    for t in parsed.turns:
        if t.role == "operator":
            continue
        if include_roles and t.role not in include_roles:
            continue
        paras = [p for p in t.paragraphs if not _BOILER_PARA_RE.search(p)]
        if paras:
            out.append("\n".join(paras))
    return "\n".join(out)


# ── header validation (F11) ──────────────────────────────────
_PERIOD_RE = re.compile(r"\bQ([1-4])\s+(?:Fiscal Year|FY)\s*(\d{4})", re.I)
_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%b. %d, %Y")
_DATE_CANDIDATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}|[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]{2,8} \d{4})")


def _parse_date(s: str):
    s = s.strip().replace("Sept", "Sep")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def validate_header(header: str, *, fiscal_quarter: str, fiscal_year: int,
                    call_date: str | None) -> list[str]:
    """Compare in-file header claims with the manifest. Returns mismatch messages."""
    problems: list[str] = []
    m = _PERIOD_RE.search(header)
    if m:
        hq, hy = f"Q{m.group(1)}", int(m.group(2))
        if (hq, hy) != (fiscal_quarter, int(fiscal_year)):
            problems.append(f"header says {hq} FY{hy} but manifest says {fiscal_quarter} FY{fiscal_year}")
    if call_date:
        found = [_parse_date(x) for x in _DATE_CANDIDATE_RE.findall(header)]
        found = [d for d in found if d]
        want = _parse_date(call_date)
        if found and want and want not in found:
            problems.append(f"header date {found[0].isoformat()} differs from manifest call_date {call_date}")
    return problems