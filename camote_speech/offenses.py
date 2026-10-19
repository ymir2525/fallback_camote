"""Full names of the offenses, as written on the LTO offense sheet (CAMOTE\\LTO_offenses.jpg). The matcher works with short labels
("Helmet"); anything shown to a person or sent to the app should use full_name(label) -> "1H - No helmet, no ICC sticker".
Standard library only."""

OFFENSES = {
    "License": ("1A / 1I", "Driving without DL, with invalid / fake / expired DL; failure to carry or surrender DL"),
    "Influence": ("1D", "Driving under the influence of alcohol / drugs"),
    "Reckless": ("1E", "Reckless driving, unsafe load, open door, swerving, accident, unsafe MV"),
    "Overspeeding": ("1E", "Over speeding"),
    "Tailgating": ("1E", "Tail gating, backing"),
    "Counterflow": ("1J.9", "Driving against the traffic, counterflow"),
    "Signs": ("1E / 1J.2", "Beating the red light; disregarding traffic signs, illegal entry"),
    "Seatbelt": ("1G.1", "Not wearing seatbelt (driver / passenger)"),
    "Helmet": ("1H", "No helmet, no ICC sticker"),
    "Registration": ("1I / 2A", "No OR/CR; unregistered MV"),
    "Parking": ("1J.1", "Illegal parking"),
    "Passenger": ("1J.3", "Excess passenger"),
    "Load": ("1J.4 / 3C", "No / insufficient canvass cover; excess load / capacity"),
    "Hitching": ("1J.5", "Hitching"),
    "Lights": ("1J.6 / 2D", "Failure to dim headlights; no / defective lights, unnecessary lights"),
    "Turn": ("1J.10", "Illegal turn, illegal U-turn"),
    "Overtaking": ("1J.11 / 1J.22", "Illegal overtaking; overtaking on no overtaking zone"),
    "Give way": ("1J.14", "Failure to give way"),
    "Obstruct": ("1J.35", "Obstruction, slow moving, overstaying, illegal loading / unloading"),
    "Modification": ("2B", "Change color / unauthorized modification / change engine"),
    "Plate": ("2E", "Tampering / no plate / improvised plate"),
    "Documents": ("4-6", "Spurious documents, fake OR/CR, plates, stickers and tags"),
    "Distracted": ("1.DD.A", "Distracted driving (RA 10913)"),
    "Children": ("1.RA 10666", "Safety of children aboard motorcycles"),
    "Discourteous": ("4-7", "Discourteous / arrogant driver"),
    "Colorum": ("4-1 / 4-5 / 1J.51", "Colorum / out of line; no CPC; breach of franchise"),
    "Overcharging": ("1J.38", "Overcharging / undercharging"),
    "Coding": ("-", "Number coding (UVVRP)"),
    "Dress": ("-", "Improper dress code (motorcycle: slippers, shorts, sleeveless)"),
}


def full_name(label):
    """'Helmet' -> '1H - No helmet, no ICC sticker' (unknown labels are returned unchanged)."""
    code, name = OFFENSES.get(label, ("", label))
    return f"{code} - {name}" if code and code != "-" else name


def full_names(labels):
    return [full_name(x) for x in labels]
