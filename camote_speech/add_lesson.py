"""Teach the phonetic fix one new mishearing:   python -m camote_speech.add_lesson "hastall" "stalled"
Adds heard -> meant to data/whisper_phonetic_db.json ("exact" table) and keeps a log in data/phonetic_lessons.json.
Only add CLEAR mistakes (not real words). Then run  python run_tests.py  to make sure nothing else changed."""
import json
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if len(sys.argv) != 3:
    sys.exit(__doc__)
heard, meant = sys.argv[1].strip().lower(), sys.argv[2].strip()
dbp, lp = os.path.join(DATA, "whisper_phonetic_db.json"), os.path.join(DATA, "phonetic_lessons.json")
db = json.load(open(dbp, encoding="utf-8"))
db["exact"][heard] = meant
json.dump(db, open(dbp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
les = json.load(open(lp, encoding="utf-8"))
les["typos"][heard] = meant
json.dump(les, open(lp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"added: {heard!r} -> {meant!r}")
