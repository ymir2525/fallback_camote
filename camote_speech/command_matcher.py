"""Phonetic COMMAND matcher: decides start_recording / scan_license / scan_cr / end_recording / capture from text that may be mis-heard.

Idea: a command = an ANCHOR word class (verb) + optionally an OBJECT word class, each matched by sound (Filipino fingerprint, small edit
distance), inside a SHORT utterance. It does not need the exact phrase, so "Styte recording", "Iska ng lisensya", "Scan mo siya" all work.
Safety: commands only fire in short utterances (word cap), never in reported speech or negation ("sabi ko mag record ka na pero...",
"dati rin akong nag scan ng license"), and an end command needs the whole utterance to be the command.
Pure standard library; ~0.1 ms per line. detect(text) -> command name or None.   python -m camote_speech.command_matcher  runs self-tests.
"""
import os
import re
import sys

sys.dont_write_bytecode = True
from .phonetic_fix import filipino_metaphone as fp  # noqa: E402


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _cls(words, maxd=None):
    """A word class: a set of fingerprints; a token is in the class if its fingerprint is within the allowed distance of any."""
    return [(fp(w), w) for w in words]


def _in(tok, cls, strict=False):
    f = fp(tok)
    if not f:
        return False
    for t, _ in cls:
        d = 0 if (len(t) <= 3 or strict) else (1 if len(t) <= 6 else 2)
        if abs(len(f) - len(t)) <= d and _lev(f, t) <= d:
            return True
    return False


SCAN = _cls(["scan", "iscan", "pascan", "paskan", "iska", "wiska", "huska", "skanda", "scand", "scam", "scanning", "iskan", "pakiscan", "pakiskan"])
LICENSE = _cls(["lisensya", "lisensiya", "license", "licence", "lisens", "laysens", "lisence", "drivers license"])
CRC = _cls(["cr", "orcr", "sier", "seer", "siar", "cier"])  # also handled as 2-token forms below
RECORD = _cls(["record", "rekord", "recording", "rekording", "regord", "regording", "magrecord", "irecord", "irekord", "rekorded"])
START = _cls(["start", "styte", "stayt", "starter", "simulan", "simalan", "begin"])
CASE = _cls(["case", "kaso", "kays", "kes", "kay"])
DONE = _cls(["done", "dun", "don", "dan", "stand", "down"], )
CAPTURE = _cls(["capture", "kapchur", "kapcher", "icapture", "pakicapture", "pacapture", "icap", "kuhanan", "kohanan", "koanan"])
PICTURE = _cls(["picture", "pic", "litrato", "larawan"])
TAPOS = _cls(["tapos", "tapus"])
STOP = _cls(["stop", "itigil", "tigil", "ihinto", "hinto", "end", "tapusin", "tapusan", "finish"])  # "stop recording", "itigil mo na yung record"
REGIS = _cls(["rehistro", "registration", "registro", "rehistrasyon", "papeles", "papers"])  # "scan registration", "scan ang rehistro"
BLOCK = {"sabi", "daw", "dati", "kanina", "hindi", "ayaw", "wag", "huwag", "nag", "nung", "noon", "kahapon", "pero", "kung", "kapag", "pag",
         "check", "chine", "ano", "anong", "bakit", "saan", "sino", "kailan", "magkano", "ilan", "paano", "ganda"}  # (question words: "Ano ba yung record ng motor na to?")


def _scan(t):  # scan-like words contain "sk" (saan, san, kanda are NOT scan)
    return _in(t, SCAN) and "sk" in fp(t)


def _cap(t):  # capture-like words are long enough not to match short everyday words (ikaw)
    return _in(t, CAPTURE) and len(fp(t)) >= 5


def tokens(text):
    t = re.sub(r"[^a-z0-9' ]", " ", (text or "").lower().replace("-", " ").replace("'", ""))
    return t.split()


def _win(toks, cls_a, cls_b, span):
    """True if a token of class A and a token of class B (different tokens) occur within `span` positions of each other."""
    ia = [i for i, t in enumerate(toks) if _in(t, cls_a)]
    ib = [i for i, t in enumerate(toks) if _in(t, cls_b)]
    return any(i != j and abs(i - j) <= span for i in ia for j in ib)


_PURPOSE = re.compile(r"\b(?:para\s+)?sa\s+record\b", re.I)  # "capture, para sa record" = purpose, not a record command
_POSSESSED = re.compile(r"\b(?:ang|yung|ng)\s+records?\s+(?:ko|mo|niya|ninyo|niyo|namin|natin|nila)\b", re.I)  # "ang record ninyo" = their record (a noun)
FILLER = ("hey", "jarvis", "okay", "ok", "sige", "boss", "sir", "po", "pa", "ng", "na", "mo", "ka", "tayo", "ang", "yung", "the", "si",
          "lang", "naman", "rin", "din", "muna", "nga", "niya", "ninyo", "niyo", "kayo")


def _detect_one(text, gate=True):
    """One command (or None) from one short utterance. gate=False skips the whole-utterance safety gate (the caller already applied it)."""
    text = _POSSESSED.sub(" ", _PURPOSE.sub(" ", text or ""))
    toks = tokens(text)
    if not toks:
        return None
    toks = [t for t in toks if t != "kasi"]  # "kasi daan" must not look like "kaso done"
    toks = [t for t in toks if t not in FILLER] or toks
    n = len(toks)
    raw_n = len(tokens(text))
    if gate and (raw_n > 12 or n > 8 or (BLOCK & set(tokens(text)) and raw_n > 3)):
        return None
    joined = "".join(toks)
    hit = []
    has_scan = any(_scan(t) for t in toks) or "skan" in fp(joined)[:6]
    has_lic = any(_in(t, LICENSE) for t in toks)
    has_cr = (any(_in(t, CRC, strict=True) for t in toks) or "".join(toks[-2:]) in ("cr", "orcr", "seear", "siyar")
              or ("see" in toks and "are" in toks) or ("c" in toks and "r" in toks) or ("or" in toks and "cr" in toks)
              or (n <= 3 and "siya" in toks and has_scan))
    has_cr = has_cr or (has_scan and any(_in(t, REGIS) for t in toks))
    stop_rec = n <= 5 and any(_in(t, STOP) for t in toks) and any(_in(t, RECORD) for t in toks)
    if has_scan and has_lic:
        hit.append("scan_license")
    elif has_scan and has_cr:
        hit.append("scan_cr")
    if stop_rec:
        hit.append("end_recording")  # "stop recording", "end recording", "itigil mo na yung record"
    elif any(_in(t, RECORD) for t in toks) and n <= 5:
        hit.append("start_recording")
    elif any(_in(t, START) for t in toks) and n <= 3 and any(t in ("recording", "record") or _in(t, RECORD) for t in toks):
        hit.append("start_recording")
    if _win(toks, CASE, DONE, 2) and n <= 4:
        hit.append("end_recording")
    elif n <= 4 and any(t in ("end", "close", "closed", "and") for t in toks[:2]) and any(_in(t, CASE) for t in toks) and not has_scan:
        hit.append("end_recording")  # "end the case", "close the case", "case closed"
    elif n <= 3 and (("tapos" in toks) or any(_in(t, TAPOS, strict=True) for t in toks)) and not has_scan and not any(_in(t, RECORD) for t in toks) and set(toks) <= {"tapos", "tapus", "kaso", "case", "done", "closed", "ito", "yan", "yun", "iyon"}:
        hit.append("end_recording")  # "tapos na" / "tapos na ang kaso" / "kaso tapos na": only when the whole utterance is that
    elif n <= 3 and "done" in toks:
        hit.append("end_recording")
    elif n <= 4 and any(t in ("tapusin", "tapusan", "isara", "isarado", "sarhan") for t in toks) and any(_in(t, CASE) for t in toks):
        hit.append("end_recording")  # "tapusin mo na yung kaso"
    cap_first = any(_cap(t) for t in toks[:2]) and n <= 8 and not (BLOCK & set(tokens(text)))  # "Capture, take a picture of the plate"
    if (any(_cap(t) for t in toks) and n <= 4) or cap_first or (n <= 4 and any(_in(t, PICTURE) for t in toks) and ("mo" in tokens(text) or n <= 2 or "take" in toks)) or ("take" in toks and n <= 6 and any(_in(t, PICTURE) for t in toks)):
        hit.append("capture")
    hit = list(dict.fromkeys(hit))
    return hit[0] if len(hit) == 1 else None


HEARSAY = {"sabi", "daw", "dati", "nung", "noon", "kahapon", "kung", "kapag", "pag", "ganda"}  # reported speech / past habit: the whole line is not an order
_SPLIT = re.compile(r"[,.;!?]+|\b(?:tapos|saka|pagkatapos|then|and)\b(?!\s+na\b)", re.I)


def detect_all(text):
    """ALL commands in an utterance, in the order spoken: "Mag-record ka na, tapos scan license" -> ["start_recording", "scan_license"].
    The whole-utterance safety gate (reported speech, questions, negation, too long) is applied first; then each clause is judged on its own."""
    whole = tokens(text)
    if not whole or len(whole) > 20:
        return []
    parts = [p for p in _SPLIT.split(text or "") if p and tokens(p)]
    if len(parts) >= 2 and not (HEARSAY & set(whole)):
        found = []  # several clauses: the safety gate judges each clause on its own ("Overspeeding ka kanina. Capture mo yung plaka.")
        for part in parts:
            c = _detect_one(part)
            if c and (not found or found[-1] != c):
                found.append(c)
        if found:
            return found
    if len(whole) > 16 or (BLOCK & set(whole) and len(whole) > 3):
        return []
    found = []
    if len(parts) >= 2:
        for part in parts:
            c = _detect_one(part, gate=False)
            if c and (not found or found[-1] != c):
                found.append(c)
        if found:
            return found
    c = _detect_one(text)
    return [c] if c else []


def detect(text):
    """The first command in the utterance, or None."""
    r = detect_all(text)
    return r[0] if r else None


if __name__ == "__main__":
    tests = {"Mag record ka na.": "start_recording", "Styte recording.": "start_recording", "Simulan mo na ang record.": "start_recording",
             "Iska ng lisensya.": "scan_license", "Boss, scan lisens na.": "scan_license", "Scan mo siya.": "scan_cr", "Scan ORCR.": "scan_cr",
             "Okay, can see y'r": None, "Case done na.": "end_recording", "Kaso tapos na.": "end_recording", "Tapos na.": "end_recording",
             "Capture mo yung plaka.": "capture", "Koanan mo ng picture.": "capture",
             "Sabi ko mag record ka na pero hindi mo ginawa": None, "Dati rin akong nag scan ng license sa opisina": None,
             "Wala kang lisensya boss": None, "End the case.": "end_recording", "Capture, take a picture of the plate.": "capture", "Na-capture namin siya kanina sa cctv": None, "Saan mo nakuha itong lisensya?": None, "Salamat po, boss. Ikaw din po, ingat.": None, "Naiwan ko po lisensya ko sa bahay boss": None, "Stop recording na.": "end_recording", "End recording.": "end_recording", "Itigil mo na yung record.": "end_recording", "Scan registration.": "scan_cr", "Scan ang rehistro.": "scan_cr", "Pakiscan ng CR.": "scan_cr", "Pakiscan ng driver's license.": "scan_license", "Tapusin mo na yung kaso.": "end_recording", "Take a picture.": "capture", "Ano ba yung record ng motor na to?": None, "Sarado po kasi yung daan, sir.": None, "Tapos na ako sa trabaho kanina pa ako dito": None}
    bad = [(k, v, detect(k)) for k, v in tests.items() if detect(k) != v]
    print("self-test:", "all passed" if not bad else bad)
