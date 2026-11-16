"""Generate a high-quality dataset of 500 balanced customer reviews with product-specific details."""

import csv
import random
from pathlib import Path

# Product definitions with tailored positive and negative aspects
PRODUCT_SPECS = [
    {
        "product": "wireless noise-canceling headphones",
        "pos": [
            "Active noise cancellation completely silences chatty coworkers and jet engine drone.",
            "Battery easily lasts over thirty hours on a single charge with ANC enabled.",
            "Memory foam ear cushions provide incredible comfort during six-hour flights.",
            "Soundstage is expansive with deep, tight bass and sparkling highs.",
            "Multipoint Bluetooth connects effortlessly between my laptop and smartphone."
        ],
        "neg": [
            "Active noise cancellation produces an uncomfortable ear-pressure sensation.",
            "The plastic headband cracked near the hinge after three weeks of gentle use.",
            "Sound is muffled and plagued by a persistent background hiss.",
            "Bluetooth disconnects repeatedly whenever my phone is in my pocket.",
            "The ear cups cause unbearable pressure and heat buildup after thirty minutes."
        ]
    },
    {
        "product": "mechanical gaming keyboard",
        "pos": [
            "Tactile switches deliver a deeply satisfying typing feel and rapid actuation.",
            "Solid aluminum backplate gives the keyboard zero flex and immense durability.",
            "Per-key RGB lighting is vivid and easily customized through the software.",
            "Detachable braided USB-C cable makes packing and travel very convenient.",
            "PBT doubleshot keycaps resist oil and shine even after months of daily coding."
        ],
        "neg": [
            "Several keys started chattering and registering double-clicks within two weeks.",
            "The spacebar stabilizer rattles loudly and feels horribly mushy.",
            "Companion software crashes frequently and fails to save custom lighting profiles.",
            "Keycap legends started wearing off after less than a month of typing.",
            "Loud pinging spring noise echoes inside the hollow plastic chassis with every keystroke."
        ]
    },
    {
        "product": "ultrawide productivity monitor",
        "pos": [
            "The massive screen real estate completely replaced my messy dual-monitor setup.",
            "IPS panel delivers vibrant color reproduction and razor-sharp text clarity.",
            "USB-C 90W power delivery charges my laptop while transmitting 4K video.",
            "The gentle curvature prevents eye fatigue when scanning wide spreadsheets.",
            "Sturdy height-adjustable stand glides smoothly and stays firmly in position."
        ],
        "neg": [
            "Arrived with four stuck pixels near the center and severe backlight bleed in corners.",
            "Screen randomly flickers black for two seconds every hour over DisplayPort.",
            "The built-in speakers sound worse than a cheap smartphone on speakerphone.",
            "HDR mode is completely washed out and looks significantly worse than SDR.",
            "The stand wobbles excessively whenever I type on my desk."
        ]
    },
    {
        "product": "espresso coffee maker",
        "pos": [
            "Extracts rich, balanced espresso with a thick golden layer of crema every morning.",
            "Powerful steam wand froths silky microfoam perfect for latte art.",
            "Heats up and reaches optimal brewing temperature in less than 45 seconds.",
            "Compact footprint fits neatly on my countertop without crowding other appliances.",
            "Intuitive pressure gauge helps dial in grind size and tamp consistency easily."
        ],
        "neg": [
            "Water pump lost pressure after three weeks and now only drips lukewarm water.",
            "Portafilter leaks pressurized coffee grounds around the gasket onto the counter.",
            "Steam wand takes five minutes to heat up and produces mostly watery foam.",
            "Descaling warning lights won't turn off even after following official maintenance instructions.",
            "Water reservoir cracked and leaked all over my wooden cabinet overnight."
        ]
    },
    {
        "product": "digital air fryer",
        "pos": [
            "Cooks french fries, wings, and veggies to crispy perfection with almost no oil.",
            "Non-stick ceramic basket wipes clean effortlessly with a damp sponge.",
            "Preheats in under two minutes and cuts cooking times in half compared to an oven.",
            "Pre-programmed presets for chicken, fish, and roasting take out all the guesswork.",
            "Runs surprisingly quiet and keeps the kitchen cool during hot summer months."
        ],
        "neg": [
            "The non-stick basket coating began flaking off into food after four gentle hand washes.",
            "Emits a sickening burnt plastic smell that persisted despite multiple burn-in cycles.",
            "Heating element failed completely on day twenty-two, leaving food raw.",
            "The touch control panel is unresponsive and requires multiple firm presses.",
            "The fan makes a deafening grinding noise every time the basket is inserted."
        ]
    },
    {
        "product": "high-speed countertop blender",
        "pos": [
            "Pulverizes frozen berries, ice cubes, and tough kale into silky smoothies in seconds.",
            "Sturdy glass pitcher feels durable and does not absorb pungent food odors.",
            "Variable speed dial and pulse function give total control over salsa texture.",
            "Self-cleaning cycle cleans the blades thoroughly with warm water and a drop of soap.",
            "1200-watt motor powers through dense nut butters without stalling or smoking."
        ],
        "neg": [
            "The motor overheated and started smoking when blending standard frozen fruit.",
            "The rubber gasket shredded, allowing liquids to leak into the motor base.",
            "Loud, ear-splitting whine makes conversation impossible during operation.",
            "Blades dulled rapidly and now leave unblended chunks of frozen fruit at the bottom.",
            "Plastic pitcher developed spiderweb stress fractures after less than a month."
        ]
    },
    {
        "product": "cordless stick vacuum",
        "pos": [
            "Featherlight design makes cleaning stairs, baseboards, and ceilings effortless.",
            "Motorized brush head effortlessly lifts stubborn pet hair from plush carpets.",
            "Swappable lithium battery provides forty minutes of consistent, fade-free suction.",
            "Bright LED headlights illuminate hidden dust and crumbs under low furniture.",
            "One-touch dustbin release empties debris cleanly directly into the trash."
        ],
        "neg": [
            "Battery dies in less than seven minutes when using boost suction mode.",
            "Brush roller clogs and stops spinning after picking up minimal pet hair.",
            "The plastic latch broke off the dustbin, causing it to fall open while vacuuming.",
            "Suction is too weak to pick up cat litter, just spitting it backwards across hard floors.",
            "The charging wall dock stopped recognizing the vacuum after only one month."
        ]
    },
    {
        "product": "waterproof hiking boots",
        "pos": [
            "Kept my socks completely bone-dry while trekking through muddy streams and wet snow.",
            "Vibram rubber outsoles deliver confident, slip-free traction on slick granite rocks.",
            "Zero break-in period required; comfortable and blister-free straight from the box.",
            "Generous ankle collar provides solid stability when carrying a 35-pound backpack.",
            "Breathable waterproof membrane prevents sweat buildup on warm summer ascents."
        ],
        "neg": [
            "Water soaked right through the toe seam during my first walk in damp grass.",
            "The metal lace eyelet ripped completely out of the leather upper on day four.",
            "The stiff heel counter caused excruciating blisters after barely three miles.",
            "Tread wore smooth with chunks of rubber tearing away after thirty trail miles.",
            "The insole is as flat and hard as plywood, offering zero arch support."
        ]
    },
    {
        "product": "cushioned running shoes",
        "pos": [
            "Pillow-soft midsole foam absorbs road impact and completely eliminated my knee ache.",
            "Breathable engineered mesh upper keeps feet cool and dry on ten-mile summer runs.",
            "Rocker sole geometry promotes a smooth, effortless toe-off transition.",
            "Roomy toe box allows natural toe splay without pinching or black toenails.",
            "Durable rubber outsole shows minimal wear after over 200 miles on asphalt."
        ],
        "neg": [
            "The foam midsole packed out and lost all cushioning after barely fifty miles.",
            "The heel collar rubs aggressively against the Achilles tendon, causing bloody blisters.",
            "Tread separated from the foam sole along the outer edge after two weeks.",
            "The upper mesh tore open near the pinky toe seam under normal running stress.",
            "Soles turn shockingly slippery on damp sidewalk pavement and painted lines."
        ]
    },
    {
        "product": "smart video doorbell",
        "pos": [
            "Crisp 2K HDR video and wide field of view clearly show packages on the doorstep.",
            "Instant motion notifications ping my phone before delivery drivers reach the porch.",
            "Two-way audio is loud and clear with effective wind noise cancellation.",
            "Night vision provides clear, recognizable facial detail in total darkness.",
            "Local micro-SD card storage means zero monthly subscription fees to view clips."
        ],
        "neg": [
            "Severe notification delay means delivery drivers have left before the video stream loads.",
            "Camera goes offline repeatedly despite having a strong -50dBm Wi-Fi signal.",
            "Motion alerts are constantly triggered by passing shadows while missing actual people.",
            "Battery requires recharging every ten days despite advertised six-month lifespan.",
            "Key features like recording and person detection are held hostage behind a paid subscription."
        ]
    },
    {
        "product": "air purifier with HEPA filter",
        "pos": [
            "Noticeably reduced morning sneezing and pet dander allergy flare-ups within 24 hours.",
            "Runs whisper-quiet on sleep mode and turns off all status lights for darkness.",
            "Real-time air quality indicator accurately senses cooking smoke and adjusts fan speed.",
            "Activated carbon filter eliminated lingering kitchen grease and onion odors quickly.",
            "Compact cylindrical form factor tucks inconspicuously into the corner of the bedroom."
        ],
        "neg": [
            "Emits a foul plastic chemical smell that gives everyone in the house a headache.",
            "The fan developed an irritating high-pitched rattle on low speed that prevents sleep.",
            "Air quality sensor is completely blind and stays on 'good' even in thick cooking smoke.",
            "Official replacement filters are exorbitantly priced and consistently out of stock.",
            "Touch controls stopped responding after three weeks, leaving the unit stuck on high."
        ]
    },
    {
        "product": "portable Bluetooth speaker",
        "pos": [
            "Shockingly big sound and punchy bass from a speaker that easily fits in a cupholder.",
            "IP67 waterproof rating handled splashing poolside and beach sand without an issue.",
            "Battery reliably powers sixteen hours of music playback at barbecue volume.",
            "Integrated bungee strap makes it effortless to clip onto backpacks or bicycles.",
            "Party-link mode paired seamlessly with my second speaker for true stereo sound."
        ],
        "neg": [
            "Audio crackles and distorts terribly whenever the volume exceeds 60 percent.",
            "Charging port became loose and now requires holding the cable at an angle to charge.",
            "Bluetooth range is pitiful; audio cuts out if you step more than ten feet away.",
            "Bass is virtually non-existent, producing a hollow, tinny sound like a cheap clock radio.",
            "Died completely after brief contact with a splash of fresh pool water."
        ]
    },
    {
        "product": "ergonomic vertical mouse",
        "pos": [
            "The handshake grip posture completely relieved my carpal tunnel and forearm strain.",
            "Whisper-quiet switch clicks are perfect for shared office and library environments.",
            "Smooth optical tracking works accurately on wood desks without requiring a mousepad.",
            "Rechargeable battery lasts nearly three months between quick USB-C charges.",
            "Convenient forward and backward thumb buttons speed up web browsing workflows."
        ],
        "neg": [
            "The optical sensor stutters and teleports across the screen during precise editing.",
            "Rubberized coating turned sticky and started peeling off after one month of daily use.",
            "The scroll wheel began squeaking loudly and frequently scrolls in the wrong direction.",
            "Too bulky for medium-sized hands; causes severe pinky finger cramping.",
            "Wireless dongle disconnects intermittently, causing frustrating cursor freezes."
        ]
    },
    {
        "product": "insulated stainless steel water bottle",
        "pos": [
            "Kept ice cubes frozen solid for over thirty hours during a hot summer road trip.",
            "Triple-lock spout lid is 100 percent leakproof even when tossed sideways in a backpack.",
            "Powder-coated textured exterior provides a firm, slip-free grip and resists scratches.",
            "Wide mouth accommodates full-size ice cubes and makes bottle cleaning a breeze.",
            "Pure 18/8 stainless steel imparts zero metallic aftertaste to pure water."
        ],
        "neg": [
            "The silicone seal leaked ice water all over my laptop inside my commuter bag.",
            "Dents extremely easily; dropping it once from table height warped the bottom.",
            "The flip lid mechanism snapped off on day three when drinking at the gym.",
            "Leaves a persistent metallic chemical taste that multiple washings could not remove.",
            "The exterior paint bubbled and peeled away after a single top-rack dishwasher cycle."
        ]
    },
    {
        "product": "electric gooseneck kettle",
        "pos": [
            "Precision gooseneck spout allows delicate, uniform water flow for pour-over coffee.",
            "Digital base holds water at exact desired temperature for up to sixty minutes.",
            "Brings one liter of cold tap water to a rolling boil in under four minutes.",
            "Sleek matte finish and minimalist aesthetic look gorgeous on our coffee bar.",
            "Balanced ergonomic handle keeps hands safe and comfortable when pouring slowly."
        ],
        "neg": [
            "Boiling water repeatedly spits and leaks from around the base onto the electrical pad.",
            "Internal temperature sensor is wildly inaccurate, overheating coffee water to a boil.",
            "Rust spots developed along the interior bottom seam within two weeks of light use.",
            "The base display started glitching and cycling random error codes after a month.",
            "Lid handle gets scorching hot to the touch, posing a serious burn hazard."
        ]
    }
]

POS_INTROS = [
    "I've been using this {product} every single day and it has exceeded all my expectations.",
    "Best purchase I've made all year; the {product} is an absolute triumph.",
    "Five stars without hesitation! This {product} delivers flawless performance.",
    "So glad I invested in this {product} instead of cheaper alternatives.",
    "Outstanding craftsmanship and design on this {product}.",
    "Everything about this {product} feels premium, durable, and well-engineered.",
    "Could not be happier with this {product} purchase.",
    "A total game changer for my routine; this {product} is worth every single penny."
]

POS_OUTROS = [
    "I would enthusiastically recommend it to anyone looking for top quality.",
    "Reliable, beautifully made, and worth every dollar.",
    "It has earned a permanent spot in my everyday setup.",
    "Customer service was also responsive and helpful when I had an initial question.",
    "Will definitely purchase more gear from this brand in the future.",
    "Exceptional quality that easily justifies the premium price tag."
]

NEG_INTROS = [
    "Extremely disappointed with this {product}; save your money and look elsewhere.",
    "Do not waste your hard-earned money on this {product}.",
    "One star is overly generous for this unreliable and defective {product}.",
    "Worst purchase I've made in years; this {product} failed almost immediately.",
    "Buyer beware: this {product} has serious design and quality control flaws.",
    "Frustrating, flimsy, and overpriced {product}.",
    "The positive reviews for this {product} must be paid because my experience was awful.",
    "Massive letdown; this {product} stopped functioning after barely any use."
]

NEG_OUTROS = [
    "I am returning it immediately and demanding a full refund.",
    "Customer support ignored my emails and refused to honor the manufacturer warranty.",
    "Avoid this company and product at all costs.",
    "Save yourself the headache and purchase from a reputable brand instead.",
    "Completely unacceptable build quality for a product in this price tier.",
    "I regret buying this and will never purchase from this brand again."
]


def generate_balanced_reviews(target_count: int = 500) -> list[dict[str, str]]:
    assert target_count % 2 == 0
    half = target_count // 2
    rng = random.Random(20261113)

    pos_reviews: list[str] = []
    seen: set[str] = set()

    # Generate 250 positive reviews
    while len(pos_reviews) < half:
        spec = rng.choice(PRODUCT_SPECS)
        prod = spec["product"]
        intro = rng.choice(POS_INTROS).format(product=prod)
        aspect = rng.choice(spec["pos"])
        outro = rng.choice(POS_OUTROS)

        style = rng.randint(1, 3)
        if style == 1:
            text = f"{intro} {aspect} {outro}"
        elif style == 2:
            text = f"{aspect} {intro}"
        else:
            text = f"{intro} {aspect}"

        if text not in seen:
            seen.add(text)
            pos_reviews.append(text)

    # Generate 250 negative reviews
    neg_reviews: list[str] = []
    while len(neg_reviews) < half:
        spec = rng.choice(PRODUCT_SPECS)
        prod = spec["product"]
        intro = rng.choice(NEG_INTROS).format(product=prod)
        aspect = rng.choice(spec["neg"])
        outro = rng.choice(NEG_OUTROS)

        style = rng.randint(1, 3)
        if style == 1:
            text = f"{intro} {aspect} {outro}"
        elif style == 2:
            text = f"{aspect} {intro}"
        else:
            text = f"{intro} {aspect}"

        if text not in seen:
            seen.add(text)
            neg_reviews.append(text)

    # Interleave and shuffle
    combined = [(t, "positive") for t in pos_reviews] + [(t, "negative") for t in neg_reviews]
    rng.shuffle(combined)

    rows: list[dict[str, str]] = []
    for idx, (text, label) in enumerate(combined, start=1):
        rows.append({
            "review_id": str(idx),
            "text": text,
            "label": label,
        })
    return rows


def main():
    output_path = Path(__file__).resolve().parent.parent / "data" / "reviews.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_balanced_reviews(500)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "text", "label"])
        writer.writeheader()
        writer.writerows(dataset)

    pos_count = sum(1 for r in dataset if r["label"] == "positive")
    neg_count = sum(1 for r in dataset if r["label"] == "negative")

    print(f"Generated {len(dataset)} tailored reviews at {output_path}")
    print(f"Distribution: {pos_count} Positive, {neg_count} Negative")


if __name__ == "__main__":
    main()
