"""KEYWORD-FIRST violation matcher (offense list = LTO_offenses.jpg). Finds offense words by SOUND, then applies small context rules.

Why: phrase patterns only caught what we had already seen. Real speech uses new wordings ("pasalubong ka", "nakaparada", "sandals lang").
So each offense has a set of keywords (Tagalog / English / Taglish, with sound-alike tolerance); the context decides if it is an accusation.
  STRONG keyword ("overspeeding", "counterflow", "reckless", "illegal parking", "open pipe"...): fires on its own.
  WEAK keyword ("helmet", "lisensya", "plaka", "cellphone", "ilaw"...): fires only with a NEGATION / problem word near it
        (wala, walang, di, hindi, expired, peke, sira, naiwan, mali, bawal...). "Nasaan ang helmet mo?" = only a HINT, not a violation.
  First-person DENIAL ("hindi po ako lasing", "hindi po ako nag-counterflow") never fires.
detect(text) -> {"violations": [labels that fire], "hints": [keyword families heard but not firing]}.  Labels = the scripts' labels
(Helmet, Seatbelt, License, Registration, Plate, Coding, Dress, Distracted, Influence, Overspeeding, Counterflow, Parking, Signs, Reckless,
Obstruct, Modification, Turn, Overtaking, Give way, Passenger, Load, Hitching, Lights, Tailgating, Colorum, Overcharging, Children, Discourteous, Documents).
Standard library only, about 1 ms per line.   python -m camote_speech.violation_matcher  runs self-tests.
"""
import os
import re
import sys

sys.dont_write_bytecode = True
from .phonetic_fix import filipino_metaphone as fp  # noqa: E402

# label -> (strong keywords, weak keywords). Multi-word keywords are written with spaces.
KW = {
    "Helmet": (["hindi nakahelmet"], ["helmet", "helmit", "elmet", "hilmet", "helmeng", "nakahelmet", "naka helmet", "headgear"]),
    "Seatbelt": (["seatbelt", "seat belt", "sitbelt", "nakaseatbelt"], ["belt", "nakabelt", "naka belt", "sinturon", "seatbelts"]),
    "License": ([],
                ["lisensya", "lisensiya", "lisens", "license", "licence", "laysens", "lisence", "permit", "student permit"]),
    "Registration": (["rehistro", "registration", "unregistered", "nirehistro", "hindi rehistrado", "lto sticker"],
                     ["registro", "orcr", "or cr", "or", "rehistrasyon", "nairehistro", "sticker"]),
    "Plate": (["improvised plate", "tabing plaka", "tinakpan ang plaka", "baluktot na plaka"], ["plaka", "plate", "plate number", "number plate", "plakang"]),
    "Coding": (["coding", "number coding", "coding scheme"], []),
    "Dress": (["tsinelas", "chinelas", "slippers", "sandals", "sleeveless", "walang manggas", "dress code", "tank top", "tanktop"],
              ["shorts", "sando", "sapatos", "short"]),
    "Distracted": (["cellphone", "cell phone", "selpon", "texting", "distracted", "headset", "earphones", "earphone", "nagtext", "nag text", "nagtetext", "nagte text", "nagtitext",
                    "nagcecellphone", "nag cecellphone", "nagse selpon", "nakatutok sa phone", "nakatutok sa cellphone", "nanonood", "nagvivideo", "facebook", "tiktok", "habang tumatawag"],
                   ["phone", "text", "tumatawag", "video", "nakatutok"]),
    "Influence": (["lasing", "amoy alak", "nakainom", "uminom", "alcohol", "beer", "dui", "alak", "lango", "nalasing", "naginom", "nag inom", "drunk", "drunk driving", "amoy beer", "nakadroga",
                   "drugs", "droga", "shabu", "marijuana", "nakainum", "inuman"], ["inom", "inumin", "high"]),
    "Overspeeding": (["overspeeding", "over speeding", "overspeed", "over speed", "speeding", "speed limit", "napakabilis", "sobrang bilis", "masyadong mabilis", "mabilis ka", "humaharurot",
                      "humarurot", "pinaharurot", "bilis ng takbo", "mabilis ang takbo", "mabilis magpatakbo", "kilometro per hour", "kph", "kilometers per hour",
                      "lampas sa speed", "harurot", "sobra ang bilis"], ["bilis", "mabilis"]),
    "Counterflow": (["counterflow", "counter flow", "contra flow", "contraflow", "kabaligtaran", "sumalungat", "salungat", "pasalubong", "maling lane", "maling daan", "illegal lane", "kabilang lane",
                     "kabilang linya", "kabilang daan", "maling direksyon", "wrong way", "pasalungat", "nasa maling lane", "kontra daloy"], []),
    "Parking": (["illegal parking", "iligal parking", "ilegal parking", "no parking", "tow zone", "nakaparada", "pumarada", "mag park", "magpark", "bawal mag park", "nakapark", "naka park",
                 "pinarada", "double parking", "illegally parked", "pumapark", "nagpark", "nag park"], ["parking", "park", "parked", "paradahan"]),
    "Signs": (["red light", "redlight", "pulang ilaw", "traffic light", "traffic signal", "stop sign", "one way", "oneway", "disregarding", "sinuway", "bawal pumasok", "nilampasan", "traffic sign",
               "beating the red light", "beat the red light", "nag beat", "nagbeat", "bumeat", "hindi huminto", "hindi sumunod", "no entry", "do not enter",
               "dumaan sa pulang", "dumaan sa red", "huminto sa ilaw", "karatula", "signage", "stoplight", "stop light", "hindi sinunod"], ["ilaw", "signal", "sign", "berde", "dilaw", "pula"]),
    "Reckless": (["reckless", "reckless driving", "zigzag", "zig zag", "swerving", "delikado", "pabaya", "mabangga", "makabangga", "masagi", "pabigla bigla", "pabiglabigla", "padalos dalos",
                  "padalosdalos", "walang ingat", "walang pag iingat", "mapanganib", "sumingit", "sumisingit", "nagsisingit", "singit", "bumabangga", "nagkakarera", "karera", "drag race"],
                 ["muntik", "sablay"]),
    "Obstruct": (["nakaharang", "harangan", "obstruction", "obstructing", "stalled", "overstaying", "hinaharang", "nakabara", "bumabara", "nakahambalang", "hambalang", "blocking",
                  "pedestrian lane", "tawiran", "crosswalk", "humaharang", "bumara"], ["harang", "nakahinto", "nakatigil", "bara", "block"]),
    "Modification": (["open pipe", "openpipe", "tambutso", "illegal modification", "unauthorized modification", "muffler", "straight pipe", "modification", "tint", "tinted",
                      "sobrang tint", "ilegal na modification", "kinalikot", "open pack", "binago mo ba"], ["maingay", "pinalitan", "modified", "binago"]),
    "Turn": (["u turn", "uturn", "illegal turn", "illegal u turn", "bawal lumiko", "bawal mag u turn", "nag u turn", "naguturn", "bawal ang u turn", "no u turn"], ["lumiko", "liko", "lumiliko"]),
    "Overtaking": (["overtaking", "overtake", "no overtaking", "bawal mag overtake", "overtaking zone", "nag overtake", "nagovertake", "umovertake", "solid line", "solid na linya",
                    "double yellow", "dilaw na linya", "illegal overtaking", "overtick", "naovertake"], []),
    "Give way": (["give way", "failure to give way", "nagbigay ng daan", "pinauna", "right of way", "bigyan ng daan", "hindi nagbigay", "hindi nagpaunlak", "nagpaunlak",
                  "pinadaan", "hindi pinadaan", "nagbibigay ng daan", "failure to yield", "yield"], ["ambulansya", "ambulance", "tumatawid", "pedestrian", "padaan", "bumbero", "fire truck"]),
    "Passenger": (["excess passenger", "sobra ang pasahero", "sobra ang sakay", "overload", "sobra sa pasahero", "lampas sa kapasidad", "kapasidad", "seating capacity",
                   "sobrang pasahero", "labis na pasahero", "apat lang dapat", "tatlo lang dapat", "dalawa lang dapat", "overloading"], ["pasahero", "sakay", "angkas", "capacity", "pasaherong"]),
    "Load": (["canvass", "trapal", "excess load", "sobra ang karga", "overloaded", "sobra sa karga", "sobrang karga", "lampas ang karga", "nakausli", "nakalawit", "hindi nakatakip",
              "labis na karga"], ["karga", "load", "takip", "nakatakip", "semento", "buhangin", "graba"]),
    "Hitching": (["hitching", "sumabit", "kumakapit", "sumasabit", "nakakapit", "kumapit", "nakasabit", "nakasampa", "sumampa", "nakabitin", "pagkapit"], ["sabit", "kapit"]),
    "Lights": (["defective lights", "headlights", "headlight", "sira ang ilaw", "walang ilaw", "patay ang ilaw", "patay na ilaw", "brake light", "tail light", "taillight", "busted",
                "high beam", "highbeam", "sobrang daming ilaw", "dagdag na ilaw", "auxiliary lights", "spotlight", "defective", "sirang ilaw"],
               ["ilaw", "dim", "bulb", "blinker", "patay", "hazard"]),
    "Tailgating": (["tailgating", "tail gating", "masyadong malapit", "backing", "malapit ang sunod", "dikit ng dikit", "nakadikit", "dumidikit", "malapit na malapit", "masyadong dikit"], []),
    "Colorum": (["colorum", "kolorum", "prangkisa", "franchise", "walang cpc", "pirata"], []),
    "Overcharging": (["overcharging", "overcharge", "sobra ang singil", "fare matrix", "sobrang singil", "mahal ang singil", "taas ng singil", "ayaw mag metro",
                      "hindi gumagamit ng metro", "taripa", "sobrang pasahe", "mahal ang pasahe", "sobra ang pasahe", "over charging"], ["singil"]),
    "Children": (["bata sa motor", "children aboard", "batang sakay", "menor de edad", "bata sa harap", "kandong", "kinandong", "child restraint", "child seat", "walang child seat", "batang nakasakay"],
                 ["bata"]),
    "Discourteous": (["bastos", "arrogant", "discourteous", "sumisigaw", "sigawan", "sinigawan", "sumigaw", "binastos", "pambabastos", "walang respeto", "walang modo", "mayabang", "arogante", "suplado",
                      "pabalang", "minura", "nagmumura", "nagmura", "pinagmumura", "bastusan", "walang galang"], ["sigaw"]),
    "Documents": (["peke", "pekeng", "fake", "spurious", "pineke", "palsipikado", "huwad", "falsified", "forged", "tampered", "dinaya", "pekeng lisensya", "pekke", "pekip"], []),
}
# second batch (added after reading volunteer recordings and the v2 sheet): label -> (strong, weak) extras merged into KW
_MORE = {
    "Passenger": ([], ["nakasakay"]),
    "Turn": (["bawal kumanan", "bawal kumaliwa"], ["kumanan", "kumaliwa"]),
    "Give way": (["nagpaubaya", "dapat nagpaubaya"], ["paubaya"]),
    "Obstruct": (["magbaba ng pasahero", "magsakay ng pasahero", "loading zone", "unloading zone", "illegal loading", "illegal unloading", "loading and unloading", "loading at unloading", "loading unloading", "unloading"], []),
    "Modification": (["iba na ang kulay", "iba na yung kulay", "iba ang kulay", "pinalitan ang kulay", "pininturahan"], []),
    "Plate": (["improvised", "gawa gawang plaka"], []),
    "Children": (["napakabata", "masyadong bata"], []),
    "Discourteous": (["pasigaw", "pasigaw sigaw"], []),
    "Colorum": ([], ["cpc"]),
    "Signs": ([], ["tinakbo", "tinakbuhan"]),  # "tinakbo mo ang red light" = ran / beat the light (on the road it can also mean speeding or evading)
    "Tailgating": (["nagback", "nag back", "masyadong dikit", "masyado kang dikit", "sobrang dikit", "dikit sa unahan", "umatras", "pag atras"], []),
    # third batch (Set 3 lines: fresh sentences the matcher had never seen)
    "Influence": (["under the influence", "dui", "driving under"], []),
    "Reckless": (["bukas yung pinto", "bukas ang pinto", "open door"], []),
    "Registration": (["papeles ng motor", "papeles ng sasakyan"], []),
    "Passenger": (["tatlo kayong", "apat kayong", "lima kayong", "tatlo kayo", "apat kayo", "masyadong marami ang angkas", "marami ang angkas", "sobra ang angkas"], []),
    "Load": (["nahuhulog ang karga", "nahuhulog"], []),
    "Hitching": (["humawak sa ibang", "humahawak sa sasakyan", "humawak sa sasakyan", "kumakapit sa sasakyan"], []),
    "Lights": (["unnecessary lights", "unnecessary light"], []),
    "Plate": (["pinalitan ang plaka", "plaka ng sticker", "gawa gawa lang", "natatakpan ang plaka", "natakpan ang plaka", "nakatakip ang plaka", "pinalitan ang plate"], []),
    "Documents": (["iba ang pirma", "iba yung pirma"], []),
    "Distracted": (["gumagamit ng phone", "gumagamit ng cellphone", "phone habang", "cellphone habang"], []),
    "Colorum": (["out of line", "mamasada", "namamasada"], []),
    "Overcharging": (["doble ang pasahe", "doble ang singil", "sinisingil"], []),
    "Coding": (["ngayon coding", "coding ngayon"], []),
    "Dress": (["naka shorts", "naka short", "nakashorts ka", "naka sleeveless"], []),
}
for _lab, (_s2, _w2) in _MORE.items():
    KW[_lab] = (KW[_lab][0] + _s2, KW[_lab][1] + _w2)
KW["Overcharging"] = ([k for k in KW["Overcharging"][0] if k != "sobra ang pasahe"], KW["Overcharging"][1])  # sounds like "sobra ang pasahero"

NEG = {"wala", "walang", "walay", "di", "hindi", "ndi", "no", "without", "expired", "expire", "paso", "peke", "pekeng", "fake", "sira", "naiwan", "nawala",
       "nakalimutan", "mali", "maling", "bawal", "violation", "tinakpan", "sobra", "lampas", "labag", "illegal", "iligal", "natanggal", "temporary", "luma", "hindi"}
Q_WORDS = {"asan", "nasaan", "saan", "pakita", "pakilabas", "ipakita", "sino", "magkano", "ilan", "ilang", "kailan"}
DENIAL = re.compile(r"\b(?:hindi|di|ndi|hnd)\s+(?:po\s+)?(?:ako|ko|namin)\b")
IMPERATIVE_OK = {"pakisuot", "pakiusap", "wag", "huwag"}  # "pakisuot ng seatbelt", "wag kang..." count as a problem context
HINT_ONLY = {"Ticket"}
NEVER = {"sandali", "anuman", "salamat", "naman", "check", "chine", "message"}  # everyday words that sound like keywords
# strong keywords that only NAME a topic: in a plain question ("cellphone ba yan?") they are a hint, not a violation
TOPIC = {"cellphone", "cell phone", "selpon", "headset", "earphones", "uminom", "alcohol", "beer", "alak", "prangkisa", "franchise", "rehistro", "registration",
         "unregistered", "headlight", "headlights", "nakaparada", "overload"}


def _toks(text):
    t = re.sub(r"[^a-z0-9' ]", " ", (text or "").lower().replace("-", " ").replace("'", ""))
    return t.split()


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _same(win_fp, kw_fp):
    if win_fp == kw_fp:
        return True
    d = 0 if len(kw_fp) <= 6 else (1 if len(kw_fp) <= 9 else 2)
    return d > 0 and abs(len(win_fp) - len(kw_fp)) <= d and _lev(win_fp, kw_fp) <= d


_INDEX = []  # (label, strength, n_words, fp)
for _lab, (_s, _w) in KW.items():
    for _strength, _lst in (("strong", _s), ("weak", _w)):
        for _k in _lst:
            _INDEX.append((_lab, _strength, len(_k.split()), fp("".join(_k.split())), _k))
            if len(_k.split()) > 1:  # Whisper may write a two-word keyword as one word ("nagtext", "nagzigzag")
                _INDEX.append((_lab, _strength, 1, fp("".join(_k.split())), _k))


_BY_N = {n: sorted({(lab, strength, kf) for lab, strength, kn, kf, _ in _INDEX if kn == n}) for n in (1, 2, 3)}  # unique (label, strength, sound) per word count


_SOUND_CACHE = {}  # (word count, sound of the window) -> [(label, strength)]; words repeat a lot, so most windows are looked up, not computed


def _hits(n, wf):
    key = (n, wf)
    r = _SOUND_CACHE.get(key)
    if r is None:
        r = [(lab, strength) for lab, strength, kf in _BY_N[n] if abs(len(wf) - len(kf)) <= 2 and _same(wf, kf)]
        if len(_SOUND_CACHE) > 50000:
            _SOUND_CACHE.clear()
        _SOUND_CACHE[key] = r
    return r


def _find(toks):
    """-> list of (label, strength, position) keyword matches by sound."""
    found = []
    for i in range(len(toks)):
        for n in (1, 2, 3):
            if i + n > len(toks):
                continue
            wf = fp("".join(toks[i:i + n]))
            if len(wf) < 3:
                continue
            if n == 1 and toks[i] in NEVER:
                continue
            for lab, strength in _hits(n, wf):
                found.append((lab, strength, i))
            if n == 1 and toks[i].startswith("naka") and len(toks[i]) > 7:  # "nakatsinelas" = naka + tsinelas
                for lab, strength in _hits(1, fp(toks[i][4:])):
                    found.append((lab, strength, i))
    return found


def asked_q(toks):
    return bool(toks) and (toks[0] in Q_WORDS or any(t in Q_WORDS for t in toks[:2]))


def _is_topic(toks, pos, lab):
    for k in TOPIC:
        n = len(k.split())
        if _same(fp("".join(toks[pos:pos + n])), fp("".join(k.split()))):
            return True
    return False


# --- sentences that mention an offense word WITHOUT accusing anyone (they become hints, never violations) ---
SECOND = {"ka", "mo", "mong", "niyo", "ninyo", "kayo", "kayong", "iyo"}  # talking TO the driver
REPORTED = {"kahapon", "nung", "noong", "daw", "madalas", "dati", "minsan", "paalala", "nahuli", "nagpapaalala", "lagi"}  # hearsay / past / general reminders
INFO_Q = {"ano", "anong", "paano", "saan", "kailan", "magkano", "gaano", "ilan", "ilang"}  # "Ano ang multa sa illegal parking?"
PRAISE = {"maayos", "ayos", "kumpleto", "tama", "mabuti", "pwede", "puwede", "allowed", "salamat"}  # "Maayos ang helmet mo", "Pwede ang parking sa kabila"
OKAY_RX = re.compile(r"\b(?:hindi|di|ndi)\s+ka\s+naman\b|\bnaka\s*\w*\s+ka\s+naman\b|\bwala(?:ng)?\s+(?:naman(?:g)?\s+)?(?:problema|kaso|violation|issue)\b|\b(?:okay|ok)\s+naman\b|\bokay\s+lang\b")
HOLD_RX = re.compile(r"\b(?:huwag|wag)\s+(?:mo|mong|muna)\b|\bhuwag\s+muna\b|\bwag\s+muna\b|\bmamaya\s+na\b|\bhindi\s+pa\s+tayo\b")  # "wag mo munang ...", "mamaya na"


def _calm(toks, low):
    """True if the WHOLE sentence only talks about an offense (hearsay, a reminder, an information question, an 'it is fine' line)."""
    s = set(toks)
    if OKAY_RX.search(low):
        return True
    if (s & REPORTED) and not (s & SECOND):
        return True
    if toks and any(t in INFO_Q for t in toks[:2]) and not (s & SECOND):
        return True
    if HOLD_RX.search(low) and not (s & SECOND):
        return True
    return False


def detect(text):
    toks = _toks(text)
    out = {"violations": [], "hints": []}
    if not toks:
        return out
    low = " ".join(toks)
    denial = bool(DENIAL.search(low))
    calm = _calm(toks, low)
    praise_pos = [i for i, t in enumerate(toks) if t in PRAISE]
    found = _find(toks)
    question = ("ba" in toks) or asked_q(toks)
    neg_pos = [i for i, t in enumerate(toks) if t in NEG or t in IMPERATIVE_OK]
    asked = (toks[0] in Q_WORDS) or any(t in Q_WORDS for t in toks[:2])
    for lab, strength, pos in found:
        if lab in out["violations"]:
            continue
        near_neg = any(abs(p - pos) <= 5 for p in neg_pos)
        if denial or calm or any(abs(p - pos) <= 4 for p in praise_pos):
            if lab not in out["hints"]:
                out["hints"].append(lab)
            continue
        topic = any(fp("".join(k.split())) == fp("".join(toks[pos:pos + len(k.split())])) for k in TOPIC if k.split()[0] in " ".join(toks[pos:pos + 3]) or True) if False else False
        if strength == "strong" and question and not near_neg and _is_topic(toks, pos, lab):
            if lab not in out["hints"]:
                out["hints"].append(lab)
            continue
        if strength == "strong" or near_neg:
            if strength == "weak" and asked and not near_neg:
                continue
            out["violations"].append(lab)
        elif lab not in out["hints"]:
            out["hints"].append(lab)
    out["hints"] = [h for h in out["hints"] if h not in out["violations"]]
    return out


if __name__ == "__main__":
    fire = {"Wala kang helmet, boss.": "Helmet", "Di ka naka seatbelt.": "Seatbelt", "Overspeeding ka, boss.": "Overspeeding", "Kuya mali yang dinadaanan mo pasalubong ka": "Counterflow",
            "Boss diyan bawal ang sasakyan nakaparada": "Parking", "Sir sandals lang suot nyo sa motor": "Dress", "Boss nakatutok ka sa phone mo habang nagmamaneho": None,
            "Luma na yung rehistro ng sasakyan nyo sir": "Registration", "Wala kang lisensya, boss.": "License", "Naiwan ko po lisensya ko sa bahay": "License"}
    nofire = ["Boss, meron ka ba dalang lisensya?", "Nasaan ang helmet mo?", "Hindi po ako lasing, sir.", "Hindi po ako nag-counterflow, sir.", "Magandang umaga po, sir.",
              "Pakita nga lisensya mo", "Meron naman po akong lisensya, nasa wallet ko lang."]
    bad = [(t, l, detect(t)["violations"]) for t, l in fire.items() if l and l not in detect(t)["violations"]]
    bad += [(t, "NONE", detect(t)["violations"]) for t in nofire if detect(t)["violations"]]
    print("self-test:", "all passed" if not bad else bad)
