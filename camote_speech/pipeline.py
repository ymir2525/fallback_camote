"""One call: raw Whisper text in, everything the backend needs out. Pure standard library, about 1-3 ms per line.

    from camote_speech import analyze
    analyze("Hastall na po, Paskan ng CR")
    -> {"raw": ..., "text": <phonetically fixed text>, "command": "scan_cr" | None, "violations": ["Helmet", ...], "hints": [...], "wake_text": False}

Order (same as the tests that measured it):  phonetic fix  ->  command matcher (fixed text first, then the raw text)  ->  violation matcher (fixed text).
`command` is the first command heard: start_recording, scan_license, scan_cr, end_recording, capture, or None. `commands` is the list of ALL commands in the
line in spoken order ("Mag-record ka na, tapos scan license" -> two); run them in that order.
`violations` are offense labels that fire; `hints` are offense words that were heard but are NOT accusations (e.g. a question) - show them to the
enforcer at most, never record them as violations.
"""
import re

from . import command_matcher, violation_matcher
from .offenses import full_names
from .phonetic_fix import PhoneticFixer, filipino_metaphone

_FIXER = None


def _fixer():
    global _FIXER
    if _FIXER is None:
        _FIXER = PhoneticFixer(valid_words=None, fuzzy=0)
    return _FIXER


def wake_in_text(text):
    """True if the TEXT contains something that sounds like 'hey jarvis' (a hint only; the real wake word is detected on the audio)."""
    toks = re.findall(r"[a-z']+", (text or "").lower())
    return any(filipino_metaphone(t) in ("jarvis", "jarbis", "harvis", "yarbis", "jarpis") for t in toks)


def analyze(text):
    raw = text or ""
    try:
        fixed = _fixer().fix(raw)
    except Exception:  # the fix must never break the pipeline
        fixed = raw
    commands = command_matcher.detect_all(fixed) or command_matcher.detect_all(raw)
    v = violation_matcher.detect(fixed)
    if any(c in ("scan_license", "scan_cr") for c in commands):
        # "Scan mo yung registration niya" only NAMES the paper to scan; it is an accusation only when a problem word is in the line ("peke", "expired", "wala")
        if not (set(re.findall(r"[a-z']+", fixed.lower())) & violation_matcher.NEG):
            v = {"violations": [x for x in v["violations"] if x not in ("License", "Registration", "Documents")],
                 "hints": v["hints"] + [x for x in v["violations"] if x in ("License", "Registration", "Documents")]}
    return {"raw": raw, "text": fixed, "command": commands[0] if commands else None, "commands": commands,
            "violations": v["violations"], "violation_names": full_names(v["violations"]), "hints": v["hints"], "wake_text": wake_in_text(raw)}
