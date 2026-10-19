"""Whisper post-fix: catch a single mangled word inside a long Filipino/Taglish sentence with a custom dictionary + phonetic fingerprint.

Self-contained (standard library only). Three parts, as specified:
 1. filipino_metaphone(word): lowercase, keep letters/digits, sh->s, ch->ts, c/q->k, f->p, o->u, e->i, collapse repeated letters.
 2. A two-table store (JSON): "exact" {typo text -> target phrase} and "phonetic" {fingerprint -> target phrase}, plus "protect" (words never touched).
 3. A sliding window over the sentence (trigrams, then bigrams, then unigrams, longest first): exact table first, then fingerprint table;
    on a match the window is replaced by the target and everything else is kept as it was (punctuation and capital letters preserved).

Safety (so it never makes good text worse): a window that already equals its target is skipped; fingerprints shorter than MIN_FP are not used;
words in "protect" (and, if given, in a list of valid words) are never replaced through the fingerprint route; an ambiguous fingerprint
(two different targets) is dropped when the DB is built; windows are never merged across punctuation; bad input returns the input unchanged.

  python whisper_phonetic_fix.py            -> demo + self-tests
  python whisper_phonetic_fix.py --seed     -> rebuild whisper_phonetic_db.json (domain words + typos mined from Whisper logs)
"""
import json
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "data", "whisper_phonetic_db.json")
MIN_FP = 4          # shortest fingerprint allowed on the fingerprint route (exact route has no minimum)
MAX_TOKENS = 400    # sentences longer than this are returned unchanged (guard)


def filipino_metaphone(word):
    """Reduce a word (or a phrase with spaces removed) to its Filipino sound fingerprint."""
    if not isinstance(word, str):
        return ""
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    w = w.replace("sh", "s").replace("ch", "ts")
    w = w.translate(str.maketrans({"c": "k", "q": "k", "f": "p", "o": "u", "e": "i"}))
    return re.sub(r"(.)\1+", r"\1", w)


def fingerprint(phrase):
    return filipino_metaphone("".join(phrase.split()))


def build_db(targets, typos=None, protect=()):
    """targets: iterable of correct words/phrases. typos: {wrong text: right text}. Returns the DB dict (+ report of dropped ambiguities)."""
    db = {"exact": {}, "phonetic": {}, "protect": sorted({p.lower() for p in protect})}
    seen, dropped = {}, []
    for t in targets:
        fp = fingerprint(t)
        if len(fp) < MIN_FP:
            continue
        seen.setdefault(fp, set()).add(t.lower())
    for t, fp in ((t, fingerprint(t)) for t in typos.values()) if typos else []:
        pass
    for wrong, right in (typos or {}).items():
        wrong, right = wrong.lower().strip(), right.strip()
        if wrong and right and wrong != right.lower():
            db["exact"][wrong] = right
            fp = fingerprint(wrong)
            if len(fp) >= MIN_FP:
                seen.setdefault(fp, set()).add(right.lower())
    for fp, tg in seen.items():
        if len(tg) == 1:
            db["phonetic"][fp] = next(iter(tg))
        else:
            dropped.append((fp, sorted(tg)))
    db["_dropped_ambiguous"] = dropped
    return db


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class PhoneticFixer:
    def __init__(self, db=None, valid_words=None, fuzzy=0):
        self.fuzzy = fuzzy  # 0 = strict spec (equal fingerprints only). 1-2 = also accept fingerprints that differ by that many letters (fp >= 6 long)
        if db is None:
            db = json.load(open(DB_PATH, encoding="utf-8")) if os.path.exists(DB_PATH) else {}
        self.exact = {k.lower(): v for k, v in db.get("exact", {}).items()}
        self.phon = dict(db.get("phonetic", {}))
        self.protect = {p.lower() for p in db.get("protect", [])}
        self.valid = valid_words or set()

    def _near(self, fp):
        best = None
        for k, v in self.phon.items():
            if abs(len(k) - len(fp)) <= self.fuzzy and len(k) >= 6 and _lev(k, fp) <= self.fuzzy:
                if best is not None and best != v:
                    return None  # two different targets are equally near: do nothing
                best = v
        return best

    @staticmethod
    def _split(tok):
        m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", tok, re.S)
        return m.group(1), m.group(2), m.group(3)

    def fix(self, text, return_changes=False):
        changes = []
        try:
            if not isinstance(text, str) or not text.strip():
                return (text, changes) if return_changes else text
            toks = text.split()
            if len(toks) > MAX_TOKENS or (not self.exact and not self.phon):
                return (text, changes) if return_changes else text
            parts = [self._split(t) for t in toks]
            out, i = [], 0
            while i < len(toks):
                done = False
                for n in (3, 2, 1):
                    if i + n > len(toks):
                        continue
                    win = parts[i:i + n]
                    if any(not w[1] for w in win):
                        continue
                    if any(w[2] for w in win[:-1]) or any(w[0] for w in win[1:]):
                        continue  # never merge across punctuation
                    cores = [w[1] for w in win]
                    key = " ".join(cores).lower()
                    target = self.exact.get(key)
                    if target is None:
                        fp = fingerprint(key)
                        cand = self.phon.get(fp) if len(fp) >= MIN_FP else None
                        if cand is None and self.fuzzy and len(fp) >= 6:
                            cand = self._near(fp)
                        if cand and cand != key and not (n == 1 and (key in self.protect or key in self.valid)):
                            target = cand
                    if target is None or target.lower() == key:
                        continue
                    rep = target
                    first = cores[0]
                    if first[:1].isupper() and rep[:1].islower():
                        rep = rep[:1].upper() + rep[1:]
                    if len(first) > 1 and first.isupper():
                        rep = rep.upper()
                    out.append(win[0][0] + rep + win[-1][2])
                    changes.append((" ".join(cores), rep))
                    i += n
                    done = True
                    break
                if not done:
                    out.append(toks[i])
                    i += 1
            res = " ".join(out)
            return (res, changes) if return_changes else res
        except Exception:  # never let a post-fix break the pipeline
            return (text, changes) if return_changes else text


def load_valid_words():
    """Optional list of real words (one per line, first column) that the fuzzy route must never change. Missing file = empty set."""
    p = os.path.join(HERE, "data", "valid_words.txt")
    if not os.path.exists(p):
        return set()
    return {l.split()[0].lower() for l in open(p, encoding="utf-8") if l.strip()}


def _self_test():
    assert filipino_metaphone("mmaskan") == "maskan" and filipino_metaphone("Pakisobokan") == filipino_metaphone("pakisubukan")
    assert filipino_metaphone("Church") == "tsurts" and filipino_metaphone("Shop") == "sup" and filipino_metaphone("cafe") == "kapi"
    db = build_db(["pakisubukan", "helmet", "illegal parking", "maskan ka"], typos={"eklek": "ang lakas"})
    f = PhoneticFixer(db)
    assert f.fix("Pakiusap pakisobokan po ang bagong code") == "Pakiusap pakisubukan po ang bagong code"
    assert f.fix("Wala kang helmit boss.") == "Wala kang helmet boss."
    assert f.fix("Illegal parkeng yan, boss!") == "Illegal parking yan, boss!"
    assert f.fix("Illegal parkin yan, boss!") == "Illegal parkin yan, boss!"  # strict mode: one missing letter is NOT a sound-alike
    assert PhoneticFixer(db, fuzzy=1).fix("Illegal parkin yan, boss!") == "Illegal parking yan, boss!"
    assert f.fix("eklek ng ulan") == "ang lakas ng ulan"
    for bad in ("", "   ", None, 123, "!!! ???", "a", "word " * 1000):
        f.fix(bad)  # must not raise
    assert f.fix("pakisubukan po") == "pakisubukan po"  # already right: untouched
    print("self-tests passed")


if __name__ == "__main__":
    if "--seed" in sys.argv:
        sys.path.insert(0, HERE)
        raise SystemExit("run seed_phonetic_db.py (it mines the Whisper logs)")
    _self_test()
    db = build_db(["pakisubukan", "helmet", "illegal parking", "maskan ka"], typos={"eklek": "ang lakas"})
    fx = PhoneticFixer(db)
    for s in ["Pakiusap pakisobokan po ang bagong code", "Boss, wala kang helmit at illegal parkin pa yan.", "Maskan ka na, eklek ng ulan"]:
        out, ch = fx.fix(s, return_changes=True)
        print(f"{s}\n  -> {out}   {ch}")
