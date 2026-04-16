"""
Generate synthetic podcast ad detection training data per SYNTHETIC_DATA_GENERATION.md.
Appends to existing synthetic_train.jsonl, filling in missing segments to reach 2,000 total.
"""

import json
import random
import math
from pathlib import Path
from collections import Counter

random.seed(42)

OUTPUT_PATH = Path("project/data/synthetic_train.jsonl")

# Load existing data
existing = []
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                existing.append(json.loads(line))

existing_label_counts = Counter(s["label"] for s in existing)
existing_cat_counts = Counter(s["category"] for s in existing)
existing_episodes = set(s["metadata"]["episode_id"] for s in existing)
next_ep_num = max(int(e.replace("ep_", "")) for e in existing_episodes) + 1 if existing_episodes else 1

print(f"Existing: {len(existing)} segments, {existing_label_counts[1]} ads, {existing_label_counts[0]} non-ads")

# Target counts minus existing
AD_TARGETS = {
    "host_read_ad": 300,
    "promo_code_ad": 200,
    "preroll_ad": 100,
    "midroll_transition": 150,
    "product_testimonial_ad": 100,
    "cross_promo": 50,
    "subtle_ad": 100,
}

NONAD_TARGETS = {
    "interview": 200,
    "monologue": 150,
    "storytelling": 150,
    "technical_discussion": 100,
    "banter": 100,
    "intro_outro": 100,
    "news_recap": 100,
    "product_mention_organic": 100,
}

ad_needed = {cat: max(0, target - existing_cat_counts.get(cat, 0)) for cat, target in AD_TARGETS.items()}
nonad_needed = {cat: max(0, target - existing_cat_counts.get(cat, 0)) for cat, target in NONAD_TARGETS.items()}

print(f"Ads needed: {sum(ad_needed.values())}")
print(f"Non-ads needed: {sum(nonad_needed.values())}")

GENRES = ["tech", "true crime", "comedy", "sports", "politics",
          "health/wellness", "business", "science", "pop culture", "education"]

AD_STYLES = ["conversational", "scripted", "blended"]

# Fictional brand names
BRANDS = [
    # VPN / Security
    ("BrightPath VPN", "brightpath.com", "SECURE"),
    ("TunnelGuard VPN", "tunnelguard.com", "GUARD"),
    ("ShieldWave Security", "shieldwave.io", "SHIELD"),
    # Coffee / Tea / Beverages
    ("GroundUp Coffee", "groundup.co", "BREW"),
    ("ZenBrew Tea", "zenbrew.co", "CALM"),
    ("MorningPerk Coffee", "morningperk.com", "PERK"),
    ("SteepLeaf Tea", "steepleaf.co", "STEEP"),
    # Mattress / Sleep
    ("NovaSleep Mattress", "novasleep.com", "REST"),
    ("DriftWell Sleep Aid", "driftwell.com", "DREAM"),
    ("CloudRest Mattress", "cloudrest.com", "SNOOZE"),
    ("DeepSlumber Pillows", "deepslumber.co", "SLUMBER"),
    # Productivity / Apps
    ("PeakMind App", "peakmind.io", "MIND"),
    ("MindMap Notes", "mindmapnotes.com", "NOTES"),
    ("FocusZone App", "focuszone.io", "FOCUS"),
    ("TaskPilot", "taskpilot.com", "PILOT"),
    ("DayPlanner Pro", "dayplannerpro.com", "PLAN"),
    # Fitness / Health
    ("FitPulse Watch", "fitpulse.com", "TRACK"),
    ("VitalFuel Bars", "vitalfuel.com", "FUEL"),
    ("FlexForm Fitness", "flexform.co", "FLEX"),
    ("IronPath Supplements", "ironpath.com", "STRONG"),
    ("PureGrit Protein", "puregrit.co", "GRIT"),
    ("StrideCoach Running", "stridecoach.com", "STRIDE"),
    # Glasses / Accessories
    ("FocusFrame Glasses", "focusframe.com", "CLEAR"),
    ("LensVault Eyewear", "lensvault.com", "VISION"),
    # Shoes / Clothing
    ("TrailBlaze Shoes", "trailblaze.com", "HIKE"),
    ("UrbanSole Sneakers", "urbansole.co", "SOLE"),
    ("ThreadLine Apparel", "threadline.com", "STYLE"),
    # Hosting / Dev Tools
    ("CloudNest Hosting", "cloudnest.dev", "LAUNCH"),
    ("CodeForge IDE", "codeforge.dev", "CODE"),
    ("StackDeploy Hosting", "stackdeploy.io", "DEPLOY"),
    ("DevBridge Platform", "devbridge.dev", "BUILD"),
    # Audio / Tech
    ("SoundWave Headphones", "soundwave.audio", "LISTEN"),
    ("BoltCharge Cables", "boltcharge.co", "POWER"),
    ("EchoBase Speakers", "echobase.com", "ECHO"),
    ("PixelCraft Monitors", "pixelcraft.co", "PIXEL"),
    # Food / Meal delivery
    ("GreenPlate Meals", "greenplate.com", "FRESH"),
    ("NourishBox Meals", "nourishbox.co", "NOURISH"),
    ("PantryDrop Delivery", "pantrydrop.com", "PANTRY"),
    # Books / Education
    ("PageTurn Books", "pageturn.co", "READ"),
    ("BrainSpark Learning", "brainspark.io", "SPARK"),
    ("LinguaLeap Language", "lingualeap.com", "LEAP"),
    # Shipping / Logistics
    ("SwiftShip Delivery", "swiftship.com", "SHIP"),
    ("QuickRoute Shipping", "quickroute.co", "ROUTE"),
    # Air / Home
    ("PureAir Filters", "pureair.co", "BREATHE"),
    ("HomeNest Furniture", "homenest.com", "NEST"),
    ("BrightSpace Lighting", "brightspace.co", "LIGHT"),
    # Skincare / Personal care
    ("Luminos Skincare", "luminos.co", "GLOW"),
    ("ClearDerm Skincare", "clearderm.com", "DERM"),
    ("FreshWave Grooming", "freshwave.co", "WAVE"),
    # Finance / Fintech
    ("CoinTrail Finance", "cointrail.com", "COIN"),
    ("BudgetWise App", "budgetwise.io", "WISE"),
    ("PayStream Payments", "paystream.co", "PAY"),
    # Pet care
    ("PawPath Pet Food", "pawpath.com", "PAW"),
    ("TailWag Treats", "tailwag.co", "WAG"),
    # Travel
    ("WanderPass Travel", "wanderpass.com", "WANDER"),
    ("JetSet Luggage", "jetsetluggage.co", "JET"),
    # Miscellaneous
    ("SafeVault Storage", "safevault.com", "VAULT"),
    ("GreenLeaf Garden", "greenleaf.co", "GARDEN"),
    ("CraftBench Tools", "craftbench.com", "CRAFT"),
]

HOST_NAMES = [
    "Jake", "Sarah", "Mike", "Priya", "Alex", "Jordan", "Marcus", "Elena",
    "Chris", "Aisha", "Tyler", "Mei", "Darius", "Kate", "Sam", "Rosa",
    "Ben", "Nina", "Liam", "Fatima", "Derek", "Sonia", "Raj", "Tanya",
]

GUEST_NAMES = [
    "Dr. Chen", "Professor Williams", "Dr. Okafor", "Maria Gonzalez",
    "James Wright", "Dr. Patel", "Lisa Thompson", "Dr. Kim",
    "Robert Singh", "Amanda Foster", "Dr. Nakamura", "Steve Collins",
    "Dr. Martinez", "Emily Park", "David O'Brien", "Dr. Hassan",
    "Rachel Green", "Dr. Ivanova", "Michael Torres", "Dr. Adebayo",
]

# ─── REAL WHISPER TRANSCRIPT EXAMPLES ───
# These are actual Whisper transcriptions from YouTube podcasts.
# Key style observations:
#   - Proper punctuation (commas, periods, apostrophes)
#   - Proper capitalization (sentence starts, proper nouns)
#   - Natural speech repetitions ("my, my", "we've, we've", "there, there's")
#   - Contractions preserved ("it's", "we've", "you're", "don't")
#   - Sentence fragments at segment boundaries
#   - Whisper sometimes mishears words ("Invidia" for "Nvidia", "electrieman" for "Lex Fridman")
#   - Run-on speech with commas instead of periods
#   - No artificial filler word injection — fillers are rare and natural

REAL_AD_EXAMPLES = [
    "And in the meantime, my, my favorite answer is eliminate waste. You know, we've, we've got all that idle power. I want to evacuate it as fast as possible. Yeah, there, there's a lot of low hanging fruit here on earth. Yeah. The working utilize for the AI scaling, quick pause, quick 30 second. Thank you to our sponsors. Check them out in the description. It really is the best way to support this podcast. Go to lexfreedman.com slash sponsors.",
    "We got perplexity for curiosity driven knowledge exploration, Shopify for selling stuff online, element for electrolytes, thin for customer service AI agents, and quote for a phone system like call text contacts for your business. Choose wise and my friends. And now back to my conversation with Jensen Kwong.",
    "a fan on the receiving end of some of those video games, you bring joy to millions of people. It's awesome. Let me ask you about quests. But first, quick math and break. It's okay. Yeah. Quick 30 second, thank you to our sponsors. Check them out in the description. It really is the best way to support this podcast.",
    "you build a big mm all around it quick pause for bathroom break quick 30 second thank you to our sponsors check them out in the description it really is the best way to support this podcast go to Lexfriedman.com slash sponsors we got Finn for customer service AI agents blitzy for co-generation in large code bases better help for mental health Shopify for selling stuff online code rabbit for AI powered code review and perplexity for curiosity driven knowledge exploration choose",
    "Can you recognize these guitars from a single note? Could you recognize the Vivaan? Abs versus air Clapton? Yeah. All right. You might be right. You might be right. Quick 30 second thank you to our sponsors. Check them all in the description. It really is the best way to support this podcast. Go to lexfremen.com slash sponsors.",
]

REAL_NONAD_EXAMPLES = [
    "And so I force everybody to think about what's the first principles, the limits, the physical limits for everything before we do anything. And we test everything against that. And so that's a good frame of mind. I don't love the other methods which is continuous improvement.",
    "up that loop, like how do you fine tune that? So it's maximal fun or fun for the maximum number of people. Is it how difficult is that? It's extremely difficult. And not everybody's good at doing that.",
    "social networks that people have that's fascinating so that's one important component of serial what else can we say about the psychology what motivates them if you look at some of the famous serial killers type on D John Wayne Gacy Jeffrey Domer is there other things you could say about their psychology that motivates them",
    "Sometimes we highlight the fact that the change in nature of music and that it's the scarcity is not there. But also allows it is like every kind of music is available and so fast and so easy. It's easy to explore to commodity. It's like turning on a water faucet.",
]

PODCAST_NAMES = {
    "tech": ["The Debug Log", "Silicon Minds", "Code & Coffee", "Byte Sized", "Tech Tangent"],
    "true crime": ["Cold Trail", "Case Closed", "Dark Files", "Missing Pieces", "The Evidence Room"],
    "comedy": ["Laugh Track", "No Filter", "The Bit", "Punchline", "Off Script"],
    "sports": ["The Huddle", "Game Day", "Court Side", "Full Time", "The Playbook"],
    "politics": ["Capitol Beat", "The Debate", "Policy Pod", "Inside Politics", "The Vote"],
    "health/wellness": ["Body Mind", "The Wellness Hour", "Healthy Habits", "Mind Body Soul", "The Recovery"],
    "business": ["Market Watch", "The Startup", "Deal Flow", "Boardroom Talk", "Revenue Stream"],
    "science": ["Lab Notes", "The Discovery", "Particle Cast", "Deep Dive Science", "The Experiment"],
    "pop culture": ["Culture Vulture", "The Buzz", "Screen Talk", "Trend Watch", "Pop Cast"],
    "education": ["Learn Daily", "The Classroom", "Knowledge Drop", "Study Break", "Ed Talk"],
}


def add_speech_disfluencies(text, density=0.02):
    """Add natural speech disfluencies matching real Whisper transcript rates.

    Real data shows ~1-2% filler prevalence, not the 30-50% the old
    add_filler() was producing. This also adds natural repetitions
    (e.g. "I, I think", "the, the thing") which Whisper transcribes faithfully.

    Disfluencies are only inserted at natural pause points (after punctuation)
    to avoid breaking mid-phrase flow.
    """
    words = text.split()
    result = []
    for i, w in enumerate(words):
        result.append(w)
        # Only insert disfluencies after natural pause points
        if i > 2 and i < len(words) - 3 and w.endswith((".", ",", "?", "!")):
            if random.random() < density:
                filler = random.choices(
                    ["I mean,", "well,", "yeah.", "right,"],
                    weights=[20, 25, 30, 25],
                )[0]
                result.append(filler)
        # Natural word repetition (~0.5% chance) — "we've, we've" style
        elif random.random() < 0.005 and i > 0 and len(w) > 2:
            # Insert repetition before current word
            result.insert(-1, w.rstrip(".,!?") + ",")
    return " ".join(result)


def whisper_style(text):
    """Make text match real Whisper transcription output.

    Real Whisper output KEEPS punctuation, capitalization, and contractions.
    Artifacts include: occasional mishearings, run-on comma splices,
    number format variation, and sentence fragments at boundaries.
    """
    # Ensure proper sentence capitalization (Whisper does this)
    sentences = text.split(". ")
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    text = ". ".join(sentences)

    # Occasional comma splice instead of period (very common in real data)
    if random.random() < 0.3:
        parts = text.split(". ")
        if len(parts) > 2:
            idx = random.randint(1, len(parts) - 2)
            parts[idx] = parts[idx][0].lower() + parts[idx][1:] if parts[idx] else parts[idx]
            text = ". ".join(parts[:idx]) + ", " + ", ".join(parts[idx:])

    # Number format variation: Whisper alternates between written and numeric
    number_swaps = [
        ("fifteen", "15"), ("twenty", "20"), ("twenty five", "25"),
        ("thirty", "30"), ("fifty", "50"), ("ten", "10"),
    ]
    if random.random() < 0.25:
        swap = random.choice(number_swaps)
        if swap[0] in text:
            text = text.replace(swap[0], swap[1], 1)
        elif swap[1] in text:
            text = text.replace(swap[1], swap[0], 1)

    # Occasional slight mishearing of proper nouns (~5% chance)
    # Mimics Whisper errors like "Invidia" for "Nvidia"
    if random.random() < 0.05:
        words = text.split()
        for i, w in enumerate(words):
            if w and w[0].isupper() and len(w) > 4 and random.random() < 0.3:
                # Swap two adjacent letters
                j = random.randint(1, len(w) - 2)
                w = w[:j] + w[j+1] + w[j] + w[j+2:]
                words[i] = w
                break
        text = " ".join(words)

    return text


PADDING_PHRASES = [
    "And I think that is really the key thing here.",
    "Which is something I have been thinking about a lot.",
    "And that is not something you hear people talk about enough.",
    "I do not know if that makes sense, but it is how I think about it.",
    "That is just my take on it though.",
    "It is one of those things where once you see it, you cannot unsee it.",
    "And I think most people would agree with me on this.",
    "I think the thing that people miss is how much context matters here.",
    "And that is coming from someone who has been doing this for a while.",
    "The more I think about it, the more convinced I am.",
    "I was talking to someone about this the other day, and they made a really good point.",
    "And I think that is worth sitting with for a second.",
    "It is wild to me that more people are not paying attention to this.",
    "And the data backs this up too, if you dig into it.",
    "I just think we need to be more honest about this stuff.",
    "And so that is a good frame of mind.",
    "It is in fact one of the things that makes this so interesting.",
    "And all of a sudden, everything starts to make sense.",
    "So that is the part that I think is really worth discussing.",
    "And I think the consensus is that this is a much bigger deal than people realize.",
]


def pad_to_length(text, min_words=75, max_words=100):
    """Pad or trim text to match real transcript word count (~88 mean)."""
    words = text.split()
    target = random.randint(min_words, max_words)
    # Trim if already over target
    if len(words) > target:
        # Cut at a sentence-ish boundary near target
        for i in range(target, min(target + 15, len(words))):
            if words[i - 1].endswith((".", "?", "!")):
                return " ".join(words[:i])
        return " ".join(words[:target])
    while len(words) < target:
        phrase = random.choice(PADDING_PHRASES)
        # Prefer appending at the end (80%) or after the last sentence boundary (20%)
        # This avoids awkward mid-paragraph insertions
        insert_points = [i for i, w in enumerate(words) if w.endswith(".") or w.endswith("?")]
        if insert_points and random.random() < 0.2:
            # Insert after the last sentence boundary only
            idx = insert_points[-1] + 1
            words = words[:idx] + phrase.split() + words[idx:]
        else:
            words.extend(phrase.split())
    return " ".join(words)


def make_messy(text):
    """Make text feel like a real Whisper transcript.

    Real Whisper output has proper punctuation and capitalization.
    We add sparse natural disfluencies and Whisper-style artifacts.
    """
    text = pad_to_length(text)
    text = add_speech_disfluencies(text, density=random.uniform(0.01, 0.03))
    text = whisper_style(text)
    return text


# ─── AD SEGMENT GENERATORS ───

def gen_host_read_ad(brand, url, code, genre, host, podcast_name):
    templates = [
        f"So I want to take a second to talk about {brand}. I've been using it for about a month now and I gotta say it's pretty solid. {random.choice(['The quality is just there.', 'It just works really well.', 'I was skeptical at first but now I recommend it to everyone.'])} If you want to check it out, head to {url} slash {podcast_name.lower().replace(' ', '')} and you'll get a special deal. Seriously, go check them out.",
        f"Alright, let me tell you about something I use every day. {brand}. {random.choice(['I started using this a few weeks ago', 'My co-host turned me onto this', 'A friend recommended this to me'])} and I'm kind of hooked. The thing that sets it apart is just how {random.choice(['easy it is to use.', 'well designed everything is.', 'reliable it has been.'])} Head to {url} and use code {code} for {random.choice(['fifteen', 'twenty', 'twenty five'])} percent off your first order.",
        f"Quick break to tell you about {brand}. Now I know, I know every podcast is sponsored by something these days, but I genuinely think this one is worth your time. {random.choice(['The customer service alone is incredible.', 'I have tried so many alternatives and nothing comes close.', 'Even my partner who is super picky about this stuff loves it.'])} Go to {url} slash {podcast_name.lower().replace(' ', '')}, link is in the show notes.",
        f"This episode is brought to you by {brand}. Look, I've been in this space for a while and {brand} is one of the few products I actually stand behind. {random.choice(['They reached out to sponsor us and I said yes immediately because I was already a customer.', 'The team behind it really cares about quality.', 'It has made a real difference in my daily routine.'])} You can try it at {url} and use code {code} at checkout.",
        # Modeled after real Lex Fridman ad style: mid-conversation transition
        f"Yeah. {random.choice(['Right, right.', 'Exactly.', 'Yeah, yeah.'])} Quick pause, quick 30 second. Thank you to our sponsors. Check them out in the description. It really is the best way to support this podcast. Go to {url} slash sponsors. We have got {brand} for {random.choice(['making your life easier', 'the thing we just talked about', 'quality you can count on'])}. Use code {code} for a special deal.",
        # Sponsor list style (very common in real data)
        f"We got {brand} for {random.choice(['productivity', 'health and wellness', 'quality tools', 'everyday essentials'])}, {random.choice(['Shopify for selling stuff online, ', 'BetterHelp for mental health, ', ''])}and {random.choice(['element for electrolytes', 'perplexity for curiosity driven knowledge exploration', 'a great deal on their website'])}. Check them out in the description. {random.choice(['Choose wisely, my friends.', 'It really helps support the show.'])} And now back to our conversation.",
    ]
    return make_messy(random.choice(templates))


def gen_promo_code_ad(brand, url, code, genre, host, podcast_name):
    templates = [
        f"Alright, so {brand} is offering our listeners an exclusive deal. Go to {url} slash {podcast_name.lower().replace(' ', '')} or use promo code {code} at checkout to get {random.choice(['fifteen', 'twenty', 'twenty five'])} percent off. That is {code}, all one word, all caps. {random.choice(['Deal runs through the end of the month.', 'Limited time offer so do not sleep on it.', 'Trust me, you are gonna love it.'])}",
        f"Head to {url} and use code {code} for a special discount just for our listeners. {brand} has been one of our favorite sponsors because their product is legit good. {random.choice(['I use it every single day.', 'We have been working with them for a while now and I can vouch for them.', 'Tons of our listeners have reached out saying they love it too.'])} Again, that is code {code} at {url}.",
        f"If you have been thinking about trying {brand}, now is the time. They are running a special where you use code {code} at checkout and get {random.choice(['a free trial', 'your first month free', 'thirty percent off'])}. That is {url}, code {code}. We will have the link in the description too.",
        f"One more time, that is {url} slash {podcast_name.lower().replace(' ', '')}, use code {code} for {random.choice(['twenty', 'fifteen', 'twenty five'])} percent off your entire order. {brand} really came through for us on this deal and {random.choice(['I think you are going to love it.', 'it is one of the best products we have promoted.', 'our listeners have been raving about it.'])}",
    ]
    return make_messy(random.choice(templates))


def gen_preroll_ad(brand, url, code, genre, host, podcast_name):
    templates = [
        f"This episode of {podcast_name} is brought to you by {brand}. Visit {url} to learn more. Now on to the show.",
        f"Before we get started, a quick word from our sponsor {brand}. Check them out at {url} and use code {code} for a special offer. Alright, let us dive in.",
        f"Today's episode is sponsored by {brand}. {random.choice(['The best in the business.', 'Making your life easier one day at a time.', 'Quality you can count on.'])} Visit {url} for more. Now let us get into it.",
        f"{podcast_name} is supported by {brand}. Go to {url} slash {podcast_name.lower().replace(' ', '')} for {random.choice(['an exclusive deal', 'a free trial', 'twenty percent off'])}. Alright, here we go.",
    ]
    return make_messy(random.choice(templates))


def gen_midroll_transition(brand, url, code, genre, host, podcast_name):
    templates = [
        f"We will get back to that in just a second, but first a quick word from our sponsor. {brand} has been keeping us going through these long recording sessions. {random.choice(['Their product is genuinely excellent.', 'I cannot say enough good things about them.', 'It has become a staple in my routine.'])} Check them out at {url} and use code {code}. Alright, back to what we were saying.",
        f"Let me take a quick break to tell you about {brand}. {random.choice(['If you are anything like me,', 'For those of you who', 'I know a lot of our listeners'])} {random.choice(['struggle with this,', 'have been looking for something like this,', 'could use something to help with this,'])} {brand} is {random.choice(['the answer.', 'what you need.', 'a total game changer.'])} {url} slash {podcast_name.lower().replace(' ', '')} for the hookup.",
        # Real-style quick transition
        f"Yeah. Quick pause, quick 30 second. Thank you to our sponsors. Check them out in the description. It really is the best way to support this podcast. We have got {brand} for {random.choice(['making things easier', 'the quality stuff', 'what you need'])}. Go to {url} and use code {code}. And now back to our conversation.",
        f"Speaking of {random.choice(['things that work,', 'good stuff,', 'quality,'])} let me tell you about {brand}. {random.choice(['I have tried a lot of products in this space', 'We get pitched by sponsors all the time', 'I am pretty picky about what I recommend'])} and {brand} is one of the few that I genuinely {random.choice(['use every day.', 'recommend to friends.', 'stand behind.'])} {url}, link in the show notes.",
    ]
    return make_messy(random.choice(templates))


def gen_product_testimonial_ad(brand, url, code, genre, host, podcast_name):
    templates = [
        f"OK so story time. I was {random.choice(['traveling last week', 'working late the other night', 'dealing with a really stressful week'])} and I {random.choice(['reached for', 'pulled out', 'decided to try'])} {brand} and it {random.choice(['completely saved me', 'made all the difference', 'was exactly what I needed'])}. I'm not just saying that because they sponsor us. I genuinely {random.choice(['use this every day', 'recommend this to everyone I know', 'think this is one of the best products out there'])}. {url} code {code} for a discount.",
        f"So funny story about {brand}. I started using them before they ever sponsored the show. {random.choice(['My wife got me into it', 'I saw an ad on another podcast ironically', 'A friend recommended it'])} and when they reached out about sponsoring I was like absolutely. Because I already knew it was good. {random.choice(['The quality speaks for itself.', 'I was already a paying customer.', 'It just made sense.'])} Go to {url} if you want to try it.",
        f"Let me share a quick personal experience with {brand}. About {random.choice(['two months', 'six weeks', 'a few weeks'])} ago I started using their {random.choice(['main product', 'premium plan', 'starter kit'])} and I have noticed a real difference. {random.choice(['My sleep has improved dramatically.', 'I am way more productive now.', 'It has genuinely changed my routine for the better.'])} I know that sounds like marketing speak but I mean it. Check them out at {url}.",
    ]
    return make_messy(random.choice(templates))


def gen_cross_promo(genre, host, podcast_name):
    other_shows = [n for g, names in PODCAST_NAMES.items() if g != genre for n in names]
    other_show = random.choice(other_shows)
    templates = [
        f"Hey if you like this show you should check out {other_show}. It's on the same network and the hosts are {random.choice(['hilarious', 'super knowledgeable', 'really engaging', 'great storytellers'])}. They just did an episode about {random.choice(['something I found fascinating', 'a topic I know our audience would love', 'one of the craziest stories I have heard this year'])}. Search for {other_show} wherever you get your podcasts.",
        f"Quick shout out to our friends over at {other_show}. If you enjoy {podcast_name} I think you will really dig what they are doing. {random.choice(['Similar vibe different topics.', 'They go deep on stuff we only touch on here.', 'Great production quality and really smart hosts.'])} Check them out link in the description.",
        f"Before we move on I want to recommend {other_show} to you. It's a {random.choice(['newer show', 'show that has been around for a while', 'podcast I have been binging lately'])} and it {random.choice(['covers some really interesting ground', 'has quickly become one of my favorites', 'deserves way more listeners than it has'])}. Give it a listen and tell them we sent you.",
    ]
    return make_messy(random.choice(templates))


def gen_subtle_ad(brand, url, code, genre, host, podcast_name):
    templates = [
        f"You know what has been helping me stay on top of all this is {brand}. I started using it a couple months ago because a friend recommended it and now I literally cannot go back. The {random.choice(['focus', 'quality', 'experience'])} is just different. Anyway they have a thing where if you go to {url} slash {podcast_name.lower().replace(' ', '')} you get twenty percent off your first order. Just throwing that out there.",
        f"That reminds me. I have been using {brand} for {random.choice(['my morning routine', 'work stuff', 'personal projects'])} and it is {random.choice(['surprisingly good', 'way better than I expected', 'become kind of essential for me'])}. I think they have a deal running right now at {url} if anyone is interested. Not a hard sell just genuinely think it is worth checking out.",
        f"Speaking of {random.choice(['good tools', 'things that work', 'being productive', 'quality stuff'])}, I have been really into {brand} lately. {random.choice(['Started as a trial and now I am fully committed.', 'Did not expect to like it this much.', 'It has kind of become part of my daily routine.'])} I think there is a link in the show notes if you want to try it. {url} or something like that.",
        f"Oh also I keep forgetting to mention this but {brand} is pretty great. I know that sounds random but {random.choice(['I was using it right before we started recording', 'someone asked me about it on Twitter', 'my co-host got me into it'])} and yeah. {url} code {code} if you are curious. Anyway back to what you were saying.",
    ]
    return make_messy(random.choice(templates))


# ─── NON-AD SEGMENT GENERATORS ───

INTERVIEW_TOPICS = {
    "tech": ["AI regulation", "open source sustainability", "startup culture", "remote work", "privacy"],
    "true crime": ["cold cases", "forensic science", "wrongful convictions", "investigative journalism", "witness protection"],
    "comedy": ["improv techniques", "writing process", "bombing on stage", "comedy specials", "roasts"],
    "sports": ["training routines", "injuries and recovery", "team dynamics", "draft picks", "coaching philosophy"],
    "politics": ["campaign strategy", "polling", "policy reform", "bipartisanship", "lobbying"],
    "health/wellness": ["mental health", "nutrition", "exercise science", "sleep hygiene", "meditation"],
    "business": ["fundraising", "scaling", "leadership", "market trends", "remote teams"],
    "science": ["climate research", "space exploration", "genetics", "neuroscience", "marine biology"],
    "pop culture": ["streaming wars", "celebrity culture", "social media trends", "fashion", "music industry"],
    "education": ["online learning", "curriculum design", "student debt", "teaching methods", "EdTech"],
}


def gen_interview(genre, host, guest):
    topic = random.choice(INTERVIEW_TOPICS.get(genre, ["general topics"]))
    templates = [
        f"So when you first started looking into {topic}, what was the thing that surprised you the most? Because I think a lot of people have this surface level understanding of it, but there is so much more going on underneath. Can you walk us through what you discovered when you really dug into it?",
        f"That is fascinating. So let me ask you this. When it comes to {topic}, do you think we are heading in the right direction or are there some fundamental things that need to change? Because from the outside looking in, it seems like there are some real problems that nobody is talking about.",
        f"I want to go back to something you said earlier about {topic}. You mentioned that there was this moment where everything kind of clicked for you. Can you tell our listeners about that? Because I think a lot of people are in that same position where they are trying to figure this stuff out.",
        f"OK, so {guest}, I have to ask you about {topic} because I know you have spent a lot of time thinking about this. What do you think is the biggest misconception people have? Because I see a lot of takes online and most of them seem to miss the point entirely.",
        f"So {guest}, you have been working on {topic} for how long now? And when you first got into it, did you expect to find what you found? Because some of the data you shared with me before the show is pretty surprising.",
        f"Here is what I do not understand about {topic}, and maybe you can help me with this. Everyone keeps saying one thing, but then the data shows something completely different. How do you reconcile that? Is it just that people do not want to hear the truth, or is there something else going on?",
        f"That is a great point. And I think it connects to what we were talking about before with {topic}. {guest}, when you present this research to people, what is the most common pushback you get? Because I imagine not everyone is receptive to what you are finding.",
    ]
    return make_messy(random.choice(templates))


def gen_monologue(genre, host):
    topic = random.choice(INTERVIEW_TOPICS.get(genre, ["general topics"]))
    templates = [
        f"So I have been thinking a lot about {topic} lately and I want to share some thoughts with you. I know this is not the usual fare for this show but bear with me because I think this is important. The thing that nobody seems to be talking about is how this affects regular people. Not the experts not the pundits just normal folks trying to figure things out.",
        f"Alright so today I want to talk about something that has been on my mind for a while. {topic}. And I know we have touched on this before but I think there is a new angle here that we have not explored. So stick with me for a few minutes because I promise this is going to be worth it.",
        f"Let me paint a picture for you. Imagine you are {random.choice(['sitting at your desk', 'driving to work', 'lying in bed at night'])} and you start thinking about {topic}. And you realize that everything you thought you knew about it is kind of wrong. That is where I was last week and it sent me down this rabbit hole that I want to share with you today.",
        f"I want to get something off my chest today about {topic}. I have been reading a lot about this and talking to people and I think we need to have an honest conversation about where things stand. Because the mainstream narrative is not telling the whole story and I think our listeners deserve better than that.",
        f"Today's episode is going to be a little different. I am going to go on a bit of a rant about {topic}. And I mean that in the best way possible. I have been sitting on this for weeks and I finally have my thoughts organized enough to share them with you. So grab your coffee or whatever you drink and let us get into it.",
    ]
    return make_messy(random.choice(templates))


def gen_storytelling(genre, host):
    templates = [
        f"So this is a story I have been wanting to tell for a while. It happened about {random.choice(['two years ago', 'last summer', 'a few months back'])} and it still kind of blows my mind when I think about it. I was {random.choice(['in this small town in the middle of nowhere', 'at a conference', 'visiting family'])} and this {random.choice(['stranger', 'person', 'old friend'])} comes up to me and says something that completely changed my perspective on everything.",
        f"OK so picture this. It is {random.choice(['three AM', 'the middle of winter', 'a random Tuesday'])} and I get this {random.choice(['phone call', 'text message', 'knock on my door'])}. And I am thinking {random.choice(['who could this possibly be', 'this cannot be good', 'what is happening right now'])}. And what happens next is something I could not have predicted in a million years.",
        f"Let me tell you about the time I {random.choice(['almost got arrested', 'met the most interesting person', 'had the worst day of my life', 'discovered something incredible'])}. This was back in {random.choice(['college', 'my twenties', 'my first year at this job'])} and I was {random.choice(['way too confident for my own good', 'completely in over my head', 'not paying attention to the signs'])}. So what happened was pretty wild.",
        f"There is this story I have never told on the show before and I think now is the right time. It involves {random.choice(['a road trip gone wrong', 'a misunderstanding that got way out of hand', 'a coincidence so bizarre you would not believe it'])} and it taught me something really important about {random.choice(['trusting your gut', 'human nature', 'paying attention to details', 'not making assumptions'])}.",
        f"So I need to set the scene for this one. We are talking {random.choice(['small town Texas', 'downtown Tokyo', 'a tiny fishing village in Portugal', 'the suburbs of Chicago'])}. It is {random.choice(['pouring rain', 'dead of night', 'the hottest day of the year'])} and I am standing there with {random.choice(['no phone', 'no money', 'no idea what I am doing'])} thinking how did I get here. And the answer to that question is a whole story in itself.",
    ]
    return make_messy(random.choice(templates))


def gen_technical_discussion(genre, host):
    tech_topics = {
        "tech": ["the new React compiler", "Rust vs Go for backend services", "WebAssembly performance", "LLM fine-tuning", "database sharding strategies"],
        "true crime": ["DNA analysis techniques", "digital forensics", "cell tower triangulation", "forensic accounting", "behavioral profiling methods"],
        "comedy": ["timing and delivery mechanics", "callback structure in sets", "the psychology of laughter", "audience reading techniques", "writing room dynamics"],
        "sports": ["analytics and sabermetrics", "biomechanics of throwing", "heart rate variability training", "film study methodology", "draft value modeling"],
        "politics": ["polling methodology", "redistricting algorithms", "policy impact modeling", "campaign finance tracking", "voter turnout prediction"],
        "health/wellness": ["gut microbiome research", "sleep architecture", "HIIT vs steady state cardio", "intermittent fasting protocols", "stress hormone cycles"],
        "business": ["unit economics", "product-market fit metrics", "cohort analysis", "burn rate optimization", "pricing strategy models"],
        "science": ["CRISPR applications", "quantum computing basics", "climate modeling", "particle physics experiments", "neural plasticity research"],
        "pop culture": ["streaming algorithm impacts", "social media engagement metrics", "content creation workflows", "audience analytics", "platform monetization"],
        "education": ["spaced repetition systems", "learning management platforms", "adaptive testing", "competency-based assessment", "educational data mining"],
    }
    topic = random.choice(tech_topics.get(genre, ["general technical topics"]))
    templates = [
        f"So let me break this down because I think there is a lot of confusion around {topic}. The way it actually works is more nuanced than most people realize. First you have to understand that there are multiple layers to this. The surface level stuff is what everyone talks about but the interesting part is what happens underneath when you really start to dig into the implementation details.",
        f"True but I also think that is a double edged sword right. Because when you have {random.choice(['twenty', 'a dozen', 'too many'])} options you also have decision fatigue and you have fragmentation. Like how many times have you seen a project where they are using some {random.choice(['random library', 'outdated framework', 'tool nobody has heard of'])} that has not been updated in two years. {topic} is a perfect example of this.",
        f"OK so I spent the weekend reading about {topic} and I want to share what I learned because it is actually really interesting. The key insight is that most people approach this wrong. They think it is about {random.choice(['optimization', 'scale', 'simplicity', 'performance'])} but really it is about {random.choice(['the tradeoffs you are willing to make', 'understanding the underlying constraints', 'knowing when good enough is good enough'])}.",
        f"The thing about {topic} that people get wrong is they try to apply {random.choice(['the same patterns', 'old thinking', 'simple solutions'])} to what is fundamentally a new problem. And I see this all the time. Someone will come in and say oh just do it this way and you look at what they are proposing and it completely misses the point of why {topic} is challenging in the first place.",
        f"Let me explain something about {topic} that I think will click for a lot of people. Think of it like a {random.choice(['spectrum', 'sliding scale', 'matrix'])}. On one end you have {random.choice(['simplicity', 'speed', 'cost efficiency'])} and on the other you have {random.choice(['correctness', 'scalability', 'maintainability'])}. Where you land on that spectrum depends entirely on your use case and most people do not take the time to figure out where they actually need to be.",
    ]
    return make_messy(random.choice(templates))


def gen_banter(genre, host):
    templates = [
        f"Wait wait wait hold on. Did you just say you have never {random.choice(['seen that movie', 'eaten sushi', 'been to a concert', 'had a real bagel'])}. How is that possible. You are like a {random.choice(['fully grown adult', 'person who claims to have taste', 'human being living in the world'])} and you have never. I cannot. We need to fix this immediately.",
        f"So before we get started I have to tell you what happened to me this morning. I {random.choice(['spilled coffee all over my keyboard', 'accidentally sent a text to the wrong person', 'locked myself out of my apartment', 'walked into a glass door'])} and I am still recovering from the embarrassment. {random.choice(['My neighbor saw the whole thing.', 'There were witnesses.', 'I will never live this down.'])}",
        f"Can we talk about the fact that {random.choice(['it is already March', 'this year is flying by', 'I still have not done my taxes', 'I am still writing the wrong year on things'])}. Like where does the time go. I feel like we just started this show yesterday and now we are on episode {random.choice(['fifty', 'a hundred', 'two hundred'])} something. That is wild to me.",
        f"OK hot take incoming and I know people are going to come at me for this but {random.choice(['pineapple on pizza is fine', 'the movie was better than the book', 'morning people are not more productive they are just annoying', 'cereal is soup'])}. I said what I said. Come fight me in the comments. {random.choice(['I will die on this hill.', 'I am not taking this back.', 'And I have receipts to prove it.'])}",
        f"Oh my god I forgot to tell you. So I was at the grocery store right and this {random.choice(['kid', 'person', 'old guy'])} comes up to me and says {random.choice(['are you the podcast guy', 'I listen to your show', 'hey you are that person from the internet'])}. And I was like yeah that is me. And they said {random.choice(['cool and walked away', 'my mom loves your show which was both flattering and insulting', 'you are taller in person which is the nicest thing anyone has ever said to me'])}.",
    ]
    return make_messy(random.choice(templates))


def gen_intro_outro(genre, host, podcast_name):
    is_intro = random.random() < 0.5
    if is_intro:
        templates = [
            f"Welcome to {podcast_name} I am your host {host} and today we have got an incredible episode lined up for you. We are going to be diving into some {random.choice(['really fascinating stuff', 'topics I have been wanting to cover for weeks', 'stories that I think are going to blow your mind'])}. If you are new to the show welcome and if you are a returning listener thank you for coming back. Let us get into it.",
            f"Hey everybody welcome back to another episode of {podcast_name}. I am {host} and {random.choice(['I am so excited about today is episode', 'we have a great show for you today', 'I have been looking forward to recording this one'])}. Before we jump in quick housekeeping. If you enjoy the show please leave us a review on whatever platform you are listening on. It really does help us out. OK let us go.",
            f"What is up everyone this is {podcast_name} episode {random.choice(['one forty seven', 'two twelve', 'eighty three', 'three twenty six'])}. I am {host} and I am {random.choice(['flying solo today', 'here with the whole crew', 'joined by a special guest'])}. We have got a packed episode so I am not going to waste any time with a long intro. Let us just dive right in.",
        ]
    else:
        templates = [
            f"And that is going to do it for today's episode of {podcast_name}. Thank you so much for listening. If you enjoyed this please share it with a friend or leave us a review. You can find us on social media at {podcast_name.replace(' ', '')} on all platforms. We will be back {random.choice(['next week', 'on Thursday', 'in a few days'])} with another episode. Until then take care of yourselves.",
            f"Alright that is a wrap on this one. Thanks as always for tuning in to {podcast_name}. If you want to support the show you can join our Patreon at patreon.com slash {podcast_name.lower().replace(' ', '')}. Even a dollar a month helps us keep the lights on. I am {host} and I will catch you in the next one. Peace.",
            f"That is all the time we have for today. Big thanks to {random.choice(['our guest for coming on', 'everyone who submitted questions', 'you the listener for sticking with us'])}. If you want more {podcast_name} content check out our {random.choice(['YouTube channel', 'Discord server', 'newsletter'])} links are in the show notes. See you next time.",
        ]
    return make_messy(random.choice(templates))


def gen_news_recap(genre, host):
    templates = [
        f"Alright let us run through some of the biggest stories this week. First up {random.choice(['and this is a big one', 'something that caught my eye', 'a story that I think flew under the radar'])}. There has been a lot of discussion about {random.choice(['the latest policy changes', 'what happened at the conference', 'the report that dropped on Tuesday', 'the announcement from last week'])}. And I think the consensus is that this is {random.choice(['a much bigger deal than people realize', 'not as bad as the headlines suggest', 'going to have some real consequences down the line'])}.",
        f"OK news roundup time. {random.choice(['Buckle up because there is a lot to cover.', 'It has been a busy week.', 'Let me catch you up on what you might have missed.'])} Number one. The thing everyone is talking about. {random.choice(['I have some thoughts on this that might be controversial.', 'I think the takes online have been mostly wrong.', 'This one is more complicated than it seems.'])} Let me break it down for you.",
        f"So if you have been paying attention to the news this week you probably saw that {random.choice(['some major changes were announced', 'there was a pretty big controversy', 'a new report came out that is worth discussing'])}. I want to talk about this because I think the coverage has been {random.choice(['kind of misleading', 'missing some important context', 'focused on the wrong thing'])}. Here is what I think is actually going on.",
        f"Time for our weekly news segment. And I will be honest this week has been {random.choice(['exhausting to follow', 'surprisingly quiet', 'absolutely wild'])}. The biggest story in my opinion is {random.choice(['not the one getting the most attention', 'something that got buried under all the other headlines', 'actually pretty straightforward when you look at the facts'])}. Let me walk you through it.",
    ]
    return make_messy(random.choice(templates))


def gen_product_mention_organic(genre, host):
    real_ish_products = [
        "that new standing desk from Ikea",
        "the latest iPhone",
        "this book I have been reading called Thinking in Systems",
        "Firefox",
        "the Kindle Paperwhite",
        "Notion for project management",
        "the Air Fryer my sister got me for Christmas",
        "Duolingo",
        "the new noise canceling headphones from Sony",
        "this meal prep service my roommate uses",
        "ChatGPT for brainstorming",
        "the Peloton I bought during lockdown",
        "VS Code with the vim extension",
        "this random tea shop I found downtown",
        "the library honestly it is underrated",
        "a cheap whiteboard from Amazon",
        "an old rice cooker that has lasted me like ten years",
        "Spotify's discover weekly",
        "this yoga app I downloaded",
        "a random podcast I found about history",
    ]
    product = random.choice(real_ish_products)
    templates = [
        f"I actually just started using {product} and oh my god the difference is insane. {random.choice(['I cannot believe I waited this long.', 'My whole routine has changed.', 'It is one of those things where you wonder how you lived without it.'])} I got it because {random.choice(['a friend recommended it', 'I saw it on a random blog', 'I was just browsing and stumbled on it'])} and it has been great. Totally worth checking out if you are into that sort of thing.",
        f"Have you tried {product}. Because I started using it recently and I have to say it is {random.choice(['pretty solid', 'way better than I expected', 'exactly what I needed'])}. Not sponsored or anything I just genuinely think it is good. {random.choice(['The build quality is nice.', 'The user experience is really smooth.', 'It just does what it is supposed to do without being annoying about it.'])}",
        f"Speaking of which I have been really into {product} lately. I know that sounds random but {random.choice(['my partner got me into it', 'I saw someone mention it online', 'I have been looking for something like this for a while'])} and it is {random.choice(['become kind of a daily thing for me', 'surprisingly addictive', 'one of my favorite discoveries this year'])}. No particular reason for bringing it up I was just thinking about it.",
        f"Oh that reminds me. I switched to {product} last month and it has been a game changer. {random.choice(['The old one I was using was so bad in comparison.', 'I wish I had done this sooner.', 'Everyone keeps asking me about it.'])} Not a paid promotion or anything just a genuine recommendation from me to you.",
    ]
    return make_messy(random.choice(templates))


# ─── EPISODE GENERATION ───

def generate_episode(ep_num, genre, segments_pool):
    """Generate a complete episode with segments from the pool."""
    ep_id = f"ep_{ep_num:04d}"
    host = random.choice(HOST_NAMES)
    podcast_name = random.choice(PODCAST_NAMES[genre])
    guest = random.choice(GUEST_NAMES)

    # Episode length: 20-90 minutes
    total_duration = random.uniform(1200, 5400)

    # Determine segment count (5-15)
    num_segments = random.randint(5, 15)

    # Build episode structure
    episode_segments = []

    # ~10% chance of no ads
    no_ads = random.random() < 0.10

    # Collect available segment types
    available_ads = [(cat, count) for cat, count in segments_pool["ads"].items() if count > 0]
    available_nonads = [(cat, count) for cat, count in segments_pool["nonads"].items() if count > 0]

    if not available_ads and not available_nonads:
        return [], segments_pool

    # Plan the episode structure
    plan = []

    # Start with intro
    if available_nonads and any(c == "intro_outro" for c, _ in available_nonads):
        plan.append(("intro_outro", 0))

    # Maybe preroll
    if not no_ads and available_ads and any(c == "preroll_ad" for c, _ in available_ads) and random.random() < 0.3:
        plan.append(("preroll_ad", 1))

    # Fill middle with content and maybe midroll ads
    remaining = num_segments - len(plan) - 1  # save 1 for outro
    ad_positions = set()
    if not no_ads and remaining > 2:
        num_midroll = random.randint(1, min(3, remaining // 3))
        ad_positions = set(random.sample(range(1, remaining), min(num_midroll, remaining - 1)))

    for i in range(remaining):
        if i in ad_positions:
            # Pick an ad category
            ad_cats = [c for c, n in segments_pool["ads"].items() if n > 0 and c not in ("preroll_ad",)]
            if ad_cats:
                cat = random.choice(ad_cats)
                plan.append((cat, 1))
            else:
                # Fall back to non-ad
                nonad_cats = [c for c, n in segments_pool["nonads"].items() if n > 0 and c != "intro_outro"]
                if nonad_cats:
                    plan.append((random.choice(nonad_cats), 0))
        else:
            nonad_cats = [c for c, n in segments_pool["nonads"].items() if n > 0 and c != "intro_outro"]
            if nonad_cats:
                plan.append((random.choice(nonad_cats), 0))

    # End with outro
    if available_nonads and any(c == "intro_outro" for c, _ in available_nonads):
        plan.append(("intro_outro", 0))

    if not plan:
        return [], segments_pool

    # Generate actual segments with timestamps
    current_time = random.uniform(0, 5)  # small offset at start
    segment_duration_share = total_duration / len(plan)

    results = []
    for cat, label in plan:
        # Check we still have budget
        if label == 1 and segments_pool["ads"].get(cat, 0) <= 0:
            continue
        if label == 0 and segments_pool["nonads"].get(cat, 0) <= 0:
            continue

        # Generate text
        brand_info = random.choice(BRANDS)
        brand, url, code = brand_info

        if cat == "host_read_ad":
            text = gen_host_read_ad(brand, url, code, genre, host, podcast_name)
        elif cat == "promo_code_ad":
            text = gen_promo_code_ad(brand, url, code, genre, host, podcast_name)
        elif cat == "preroll_ad":
            text = gen_preroll_ad(brand, url, code, genre, host, podcast_name)
        elif cat == "midroll_transition":
            text = gen_midroll_transition(brand, url, code, genre, host, podcast_name)
        elif cat == "product_testimonial_ad":
            text = gen_product_testimonial_ad(brand, url, code, genre, host, podcast_name)
        elif cat == "cross_promo":
            text = gen_cross_promo(genre, host, podcast_name)
        elif cat == "subtle_ad":
            text = gen_subtle_ad(brand, url, code, genre, host, podcast_name)
        elif cat == "interview":
            text = gen_interview(genre, host, guest)
        elif cat == "monologue":
            text = gen_monologue(genre, host)
        elif cat == "storytelling":
            text = gen_storytelling(genre, host)
        elif cat == "technical_discussion":
            text = gen_technical_discussion(genre, host)
        elif cat == "banter":
            text = gen_banter(genre, host)
        elif cat == "intro_outro":
            text = gen_intro_outro(genre, host, podcast_name)
        elif cat == "news_recap":
            text = gen_news_recap(genre, host)
        elif cat == "product_mention_organic":
            text = gen_product_mention_organic(genre, host)
        else:
            continue

        # Calculate duration from word count
        word_count = len(text.split())
        duration = (word_count / 150) * 60  # ~150 wpm
        # Add some natural variance
        duration *= random.uniform(0.85, 1.15)
        duration = max(10, duration)

        start_time = round(current_time, 1)
        end_time = round(current_time + duration, 1)

        metadata = {
            "podcast_genre": genre,
            "episode_id": ep_id,
        }
        if label == 1:
            metadata["ad_style"] = random.choice(AD_STYLES)

        segment = {
            "text": text,
            "label": label,
            "category": cat,
            "start_time": start_time,
            "end_time": end_time,
            "metadata": metadata,
        }
        results.append(segment)

        # Decrement pool
        if label == 1:
            segments_pool["ads"][cat] -= 1
        else:
            segments_pool["nonads"][cat] -= 1

        # Advance time with small gap
        current_time = end_time + random.uniform(0.5, 5.0)

    return results, segments_pool


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Approx segments per batch (default: 100)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Run a single batch number (1-indexed). If omitted, generates all.")
    parser.add_argument("--shuffle-only", action="store_true",
                        help="Just shuffle the existing file (run after all batches)")
    args = parser.parse_args()

    if args.shuffle_only:
        print("Shuffling existing data...")
        all_segments = list(existing)
        random.shuffle(all_segments)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for seg in all_segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        print(f"Shuffled {len(all_segments)} segments.")
        return

    total_needed = sum(ad_needed.values()) + sum(nonad_needed.values())
    if total_needed == 0:
        print("Dataset is already complete! Nothing to generate.")
        return

    # Calculate how many batches we need
    num_batches = math.ceil(total_needed / args.batch_size)
    print(f"\nTotal segments to generate: {total_needed}")
    print(f"Batch size: ~{args.batch_size} segments")
    print(f"Estimated batches needed: {num_batches}")

    # Divide category targets across batches proportionally
    def split_targets(targets, batch_num, num_batches):
        """Get the target counts for a specific batch."""
        batch_targets = {}
        for cat, total in targets.items():
            per_batch = total / num_batches
            # Give each batch its fair share, rounding to handle remainders
            start = int(round(per_batch * (batch_num - 1)))
            end = int(round(per_batch * batch_num))
            batch_targets[cat] = end - start
        return batch_targets

    if args.batch is not None:
        batches_to_run = [args.batch]
    else:
        batches_to_run = list(range(1, num_batches + 1))

    for batch_num in batches_to_run:
        # Use batch_num as part of seed for reproducibility but variety
        random.seed(42 + batch_num * 1000)

        batch_ad_targets = split_targets(ad_needed, batch_num, num_batches)
        batch_nonad_targets = split_targets(nonad_needed, batch_num, num_batches)

        segments_pool = {
            "ads": batch_ad_targets,
            "nonads": batch_nonad_targets,
        }

        batch_total = sum(batch_ad_targets.values()) + sum(batch_nonad_targets.values())
        print(f"\n--- Batch {batch_num}/{num_batches} ({batch_total} segments) ---")

        # Determine episode number offset for this batch
        ep_offset = next_ep_num + (batch_num - 1) * 20  # reserve 20 eps per batch
        ep_num = ep_offset

        # Assign 2 genres per batch, cycling through all 10
        genre_start = ((batch_num - 1) * 2) % len(GENRES)
        batch_genres = [GENRES[genre_start % len(GENRES)], GENRES[(genre_start + 1) % len(GENRES)]]
        genre_cycle = batch_genres * 20
        genre_idx = 0

        batch_segments = []
        max_iterations = 100
        iteration = 0

        while sum(segments_pool["ads"].values()) + sum(segments_pool["nonads"].values()) > 0:
            iteration += 1
            if iteration > max_iterations:
                remaining = sum(segments_pool['ads'].values()) + sum(segments_pool['nonads'].values())
                print(f"  Warning: hit max iterations with {remaining} segments remaining")
                break

            genre = genre_cycle[genre_idx % len(genre_cycle)]
            genre_idx += 1

            new_segs, segments_pool = generate_episode(ep_num, genre, segments_pool)
            batch_segments.extend(new_segs)
            ep_num += 1

        print(f"  Generated {len(batch_segments)} segments across {ep_num - ep_offset} episodes")
        print(f"  Genres: {batch_genres}")

        # Append to file (don't shuffle yet — shuffle at the end)
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for seg in batch_segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        print(f"  Appended to {OUTPUT_PATH}")

    # Print current file stats
    all_data = []
    with open(OUTPUT_PATH) as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    labels = Counter(s["label"] for s in all_data)
    cats = Counter(s["category"] for s in all_data)
    genres = Counter(s["metadata"]["podcast_genre"] for s in all_data)
    episodes = len(set(s["metadata"]["episode_id"] for s in all_data))

    print(f"\n{'='*50}")
    print(f"Current file stats ({len(all_data)} total segments):")
    print(f"  Ads: {labels[1]}, Non-ads: {labels[0]}")
    print(f"  Episodes: {episodes}")
    print(f"  Genres: {dict(genres)}")
    print(f"  Categories:")
    for c, n in sorted(cats.items()):
        print(f"    {c}: {n}")

    remaining_ads = 1000 - labels.get(1, 0)
    remaining_nonads = 1000 - labels.get(0, 0)
    if remaining_ads > 0 or remaining_nonads > 0:
        print(f"\n  Still needed: ~{remaining_ads} ads, ~{remaining_nonads} non-ads")
        print(f"  Run more batches or run with --shuffle-only when done")
    else:
        print(f"\n  Dataset target reached! Run with --shuffle-only to finalize.")


if __name__ == "__main__":
    main()
