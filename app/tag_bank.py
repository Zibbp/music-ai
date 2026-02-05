# app/tag_bank.py

# -----------------------
# Stage 1: Family gate
# -----------------------
FAMILY_LABELS = ["electronic", "dnb", "metal", "rock_punk", "hiphop_rnb", "classical_jazz", "soundtrack_world", "other"]

FAMILY_PROMPTS = {
    "electronic": [
        "electronic music, synthesized or digital instruments, drum machines, programmed beats, electronic textures (EDM/IDM/techno/house/ambient)",
        "electronic dance music, club/rave track, synthesizers, electronic drums, dance beat (house/techno/trance/dubstep/hard dance)",
    ],
    "dnb":              "drum and bass or jungle, fast breakbeats, rolling bassline, high tempo electronic",
    "metal":            "metal music, distorted guitars, heavy riffs, aggressive drums, heavy sound",
    "rock_punk":        "rock or punk band, electric guitars, live drums, rock groove, band performance",
    "hiphop_rnb":       "hip hop or rap, beats, rapping vocals, 808 bass, trap or boom bap",
    "classical_jazz": [
        "acoustic jazz ensemble, improvisation, swing feel, saxophone trumpet piano upright bass, live drums",
        "classical or orchestral music, strings, woodwinds, brass, piano, composed, symphonic",
    ],
    "soundtrack_world": "soundtrack score or world music, cinematic or regional traditional instruments",
    "other":            "miscellaneous music that does not fit electronic, dnb, metal, rock, hip hop, classical, jazz, soundtrack, world",
}

# -----------------------
# Stage 2: per-family genres
# -----------------------

# Electronic family subgenres
# (keeps your core buckets, adds a few high-signal missing ones like psytrance + frenchcore via hard_dance)
ELECTRONIC_GENRE_LABELS = [
    "techno",
    "house",
    "trance",
    "psytrance",
    "dubstep",
    "hard_dance",
    "breaks",
    "ambient_edm",
    "edm_other",
]
ELECTRONIC_GENRE_PROMPTS = {
    "techno":      "techno track, driving four-on-the-floor kick, repetitive groove, club rave, minimal vocals (includes hard techno/industrial techno/minimal)",
    "house":       "house music, four-on-the-floor, groovy bassline, dance club track (includes deep house/progressive house/tech house/funky house)",
    "trance":      "trance, uplifting melody, arpeggiated synths, big breakdown and build-up, euphoric (includes progressive trance)",
    "psytrance":   "psytrance, fast four-on-the-floor, hypnotic rolling bass, psychedelic synth lines, festival or underground trance",
    "dubstep":     "dubstep, heavy sub bass, wobble bass, halftime groove, big drop, aggressive electronic (includes brostep/riddim/drumstep)",
    "hard_dance":  "hard dance rave, hardstyle or hardcore, pounding kick drum, very high energy, fast (includes frenchcore/gabber/uptempo/rawstyle/happy hardcore)",
    "breaks":      "breakbeat or breaks, broken drum patterns, funky breakbeats, big beat / nu-skool breaks, electronic dance track",
    "ambient_edm": "ambient electronic or downtempo, atmospheric pads, chill electronic, slow beat or no drums (includes chillout/IDM-leaning downtempo)",
    "edm_other":   "festival EDM, big synth hook, pop-leaning dance track, modern electronic (includes electro house/future bass/bass house/melodic EDM)",
}

# DnB family subgenres
# (adds dancefloor as a common missing bucket; keeps your originals)
DNB_GENRE_LABELS = ["dnb_liquid", "dnb_neuro", "dnb_jump_up", "dnb_rollers", "dnb_jungle", "dnb_dancefloor", "dnb_other"]
DNB_GENRE_PROMPTS = {
    "dnb_liquid":     "liquid drum and bass, smooth pads, melodic, rolling breakbeats, warm bass",
    "dnb_neuro":      "neurofunk drum and bass, aggressive reese bass, dark, technical, intense",
    "dnb_jump_up":    "jump up drum and bass, bouncy bassline, simple hooks, party vibe",
    "dnb_rollers":    "rollers drum and bass, rolling groove, minimal melody, steady driving bass",
    "dnb_jungle":     "jungle, chopped breakbeats, reggae influence, old school rave, fast breaks",
    "dnb_dancefloor": "dancefloor drum and bass, big melodic hooks, energetic drops, polished modern DnB",
    "dnb_other":      "drum and bass track, fast breakbeats, rolling bassline, electronic",
}

# Metal family subgenres
# (keeps structure; expands prompts so you catch common overlaps like nu/alt/sludge without adding more labels)
METAL_GENRE_LABELS = ["heavy_metal", "thrash", "death", "black", "power", "doom", "metalcore", "industrial_metal", "prog_metal", "metal_other"]
METAL_GENRE_PROMPTS = {
    "heavy_metal":       "heavy metal, classic metal riffs, powerful vocals, guitar solos, driving drums",
    "thrash":            "thrash metal, fast palm-muted riffs, aggressive, rapid drums, high energy",
    "death":             "death metal, growled vocals, blast beats, heavy distorted guitars, brutal",
    "black":             "black metal, tremolo picking, harsh vocals, dark atmosphere, fast drums",
    "power":             "power metal, uplifting epic melodies, fast double-kick, clean vocals",
    "doom":              "doom metal, slow heavy riffs, dark and heavy, thick guitars, slow tempo (includes sludge/stoner-leaning doom)",
    "metalcore":         "metalcore, breakdowns, chugging riffs, screamed vocals, modern heavy (includes post-metalcore/djent-leaning riffs)",
    "industrial_metal":  "industrial metal, mechanical rhythms, electronic elements, heavy guitars (includes aggrotech-leaning or electro-industrial metal)",
    "prog_metal":        "progressive metal, complex rhythms, technical playing, odd time signatures (includes djent/prog-heavy hybrids)",
    "metal_other":       "metal track, heavy guitars, aggressive drums, heavy sound (includes nu metal/alt metal when not clearly another bucket)",
}

# Rock/Punk family subgenres
# (adds two high-signal buckets; keeps your existing ones)
ROCK_PUNK_GENRE_LABELS = ["alt_rock", "hard_rock", "indie_rock", "punk", "hardcore_punk", "post_hardcore", "emo", "pop_punk", "grunge", "rock_other"]
ROCK_PUNK_GENRE_PROMPTS = {
    "alt_rock":       "alternative rock, modern rock band, guitar-driven, rock groove (includes post-grunge / modern alt)",
    "hard_rock":      "hard rock, loud guitar riffs, big drums, arena rock energy",
    "indie_rock":     "indie rock, jangly guitars, alternative pop sensibility, band sound",
    "punk":           "punk rock, fast simple chords, raw vocals, energetic, rebellious",
    "hardcore_punk":  "hardcore punk, very fast, aggressive shouting vocals, intense drums",
    "post_hardcore":  "post-hardcore, aggressive and melodic, dynamic loud/quiet shifts",
    "emo":            "emo rock, emotional vocals, melodic guitars, pop-punk/alt influence",
    "pop_punk":       "pop punk, upbeat punk energy, catchy hooks, power chords, punk-pop vocals",
    "grunge":         "grunge rock, gritty guitars, 90s alt feel, heavy but melodic, raw band sound",
    "rock_other":     "rock music, electric guitars, live drums, band performance",
}

# Hip hop / R&B family subgenres
# (adds drill; keeps your existing buckets)
HIPHOP_RNB_GENRE_LABELS = ["trap", "boom_bap", "drill", "hiphop_pop", "phonk", "rnb", "hiphop_other"]
HIPHOP_RNB_GENRE_PROMPTS = {
    "trap":        "trap rap, rapping vocals, 808 bass, hi-hat rolls, modern rap beat, punchy kick",
    "boom_bap":    "boom bap hip hop, rapping vocals, classic rap beat, sampled drums, old school vibe",
    "drill":       "drill rap, rapping vocals, sliding 808s, sparse dark beat, syncopated hi-hats, aggressive flow",
    "hiphop_pop":  "pop rap, rapping or melodic rap vocals, catchy hook, modern hip hop production",
    "phonk":       "phonk, cowbell, memphis rap influence, gritty, dark trap feel",
    "rnb":         "R&B, smooth vocals, soulful, groove, contemporary R&B production",
    "hiphop_other":"hip hop or rap, rapping vocals, beats, 808 bass",
}


# Classical / Jazz family subgenres
# (keeps structure; slightly strengthens edge-cases like modern classical)
CLASSICAL_JAZZ_GENRE_LABELS = ["classical", "orchestral_score", "piano_solo", "jazz", "blues", "swing", "cj_other"]
CLASSICAL_JAZZ_GENRE_PROMPTS = {
    "classical":        "classical music, orchestral instruments, composed, symphonic (includes modern classical/chamber when clearly classical)",
    "orchestral_score": "orchestral score, cinematic classical, strings brass percussion, dramatic",
    "piano_solo":       "solo piano, classical or modern classical, expressive, instrumental",
    "jazz":             "jazz ensemble, improvisation, swing feel, saxophone trumpet piano (includes bebop/cool/modern jazz)",
    "blues":            "blues, blues guitar, soulful vocals, 12-bar feel, blues groove",
    "swing":            "swing jazz, big band, upbeat swing rhythm, brass section",
    "cj_other":         "classical or jazz music, acoustic instruments, instrumental",
}

# Soundtrack / World family subgenres
SOUNDTRACK_WORLD_GENRE_LABELS = ["film_score", "game_score", "anime_score", "world", "regional_dance", "stw_other"]
SOUNDTRACK_WORLD_GENRE_PROMPTS = {
    "film_score":      "film soundtrack score, cinematic orchestration, dramatic themes",
    "game_score":      "video game soundtrack, cinematic or electronic score, background music",
    "anime_score":     "anime soundtrack, orchestral or pop-influenced score, thematic",
    "world":           "world music, traditional instruments, regional folk styles",
    "regional_dance":  "regional dance music, traditional rhythms, cultural instruments, upbeat",
    "stw_other":       "soundtrack or world music, cinematic or regional instruments",
}

# -----------------------
# Mood / energy (shared)
# -----------------------
MOOD_LABELS = ["upbeat", "energetic", "dark", "calm", "sad", "chill", "aggressive", "heavy"]

MOOD_PROMPTS = {
    "upbeat":     "upbeat, uplifting, positive energy, happy vibe",
    "energetic":  "high energy, intense, driving, fast-paced",
    "dark":       "dark, ominous, moody, intense atmosphere",
    "calm":       "calm, relaxing, mellow, gentle",
    "sad":        "sad, melancholic, emotional",
    "chill":      "chill, laid back, smooth, easy listening",
    "aggressive": "aggressive, angry, hard-hitting, harsh",
    "heavy":      "heavy, powerful, intense, thick sound",
}


# Helper mapping for stage-2 banks
STAGE2 = {
    "electronic": (ELECTRONIC_GENRE_LABELS, ELECTRONIC_GENRE_PROMPTS),
    "dnb":        (DNB_GENRE_LABELS, DNB_GENRE_PROMPTS),
    "metal":      (METAL_GENRE_LABELS, METAL_GENRE_PROMPTS),
    "rock_punk":  (ROCK_PUNK_GENRE_LABELS, ROCK_PUNK_GENRE_PROMPTS),
    "hiphop_rnb": (HIPHOP_RNB_GENRE_LABELS, HIPHOP_RNB_GENRE_PROMPTS),
    "classical_jazz": (CLASSICAL_JAZZ_GENRE_LABELS, CLASSICAL_JAZZ_GENRE_PROMPTS),
    "soundtrack_world": (SOUNDTRACK_WORLD_GENRE_LABELS, SOUNDTRACK_WORLD_GENRE_PROMPTS),
}

# Global genre bank and family mapping for cross-family selection
GENRE_ALL_LABELS = (
    ELECTRONIC_GENRE_LABELS
    + DNB_GENRE_LABELS
    + METAL_GENRE_LABELS
    + ROCK_PUNK_GENRE_LABELS
    + HIPHOP_RNB_GENRE_LABELS
    + CLASSICAL_JAZZ_GENRE_LABELS
    + SOUNDTRACK_WORLD_GENRE_LABELS
)

GENRE_ALL_PROMPTS = {}
GENRE_TO_FAMILY = {}
for _fam, (_labels, _prompts) in STAGE2.items():
    for _lab in _labels:
        GENRE_ALL_PROMPTS[_lab] = _prompts[_lab]
        GENRE_TO_FAMILY[_lab] = _fam

# Fallback genre per family when confidence is low
FAMILY_FALLBACK_GENRE = {
    "electronic": "edm_other",
    "dnb": "dnb_other",
    "metal": "metal_other",
    "rock_punk": "rock_other",
    "hiphop_rnb": "hiphop_other",
    "classical_jazz": "cj_other",
    "soundtrack_world": "stw_other",
}
