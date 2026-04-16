# Synthetic Data Generation Instructions

These instructions are for an LLM to generate synthetic training data for the **Podcast Ad Detection** project. The goal is to produce labeled text segments that mimic real podcast transcripts, classified as either **ad** or **non-ad** content.

---

## Output Format

Generate data as a JSON Lines (`.jsonl`) file. Each line is a JSON object:

```json
{"text": "the transcript segment text", "label": 1, "category": "host_read_ad", "start_time": 847.2, "end_time": 903.6, "metadata": {"podcast_genre": "tech", "ad_style": "conversational", "episode_id": "ep_0042"}}
```

- `text`: The transcript segment (3-10 sentences, roughly 50-200 words)
- `label`: `1` for ad content, `0` for non-ad content
- `category`: A subcategory tag (see below)
- `start_time`: Simulated start timestamp in seconds from the beginning of the episode (float)
- `end_time`: Simulated end timestamp in seconds from the beginning of the episode (float)
- `metadata`: Context about the segment
  - `podcast_genre`: The genre of the podcast (see genres list below)
  - `ad_style` (ads only): e.g. "conversational", "scripted", "blended"
  - `episode_id`: A shared identifier grouping segments that belong to the same simulated episode (e.g. `"ep_0042"`)

---

## Dataset Composition

Generate **2,000 total segments**: 1,000 ad segments and 1,000 non-ad segments.

### Ad Segments (label: 1) — 1,000 total

Generate across these categories with approximate counts:

| Category | Count | Description |
|---|---|---|
| `host_read_ad` | 300 | The host personally endorses a product in their own voice/style. Most common podcast ad format. |
| `promo_code_ad` | 200 | Ad that includes a specific promo code or vanity URL for listeners. |
| `preroll_ad` | 100 | Short ad at the very start of an episode, often scripted and formulaic. |
| `midroll_transition` | 150 | Ad introduced with a transition phrase like "a quick word from our sponsors" or "let me tell you about...". |
| `product_testimonial_ad` | 100 | Host shares a personal story about using the product, blending ad and genuine experience. |
| `cross_promo` | 50 | Promoting another podcast or show on the same network. |
| `subtle_ad` | 100 | Very natural-sounding plugs that are hard to distinguish from regular conversation. These are the hardest cases. |

### Non-Ad Segments (label: 0) — 1,000 total

Generate across these categories with approximate counts:

| Category | Count | Description |
|---|---|---|
| `interview` | 150 | Host interviewing a guest — questions, answers, back-and-forth. |
| `monologue` | 125 | Host speaking solo about the episode's topic. |
| `storytelling` | 125 | Narrative segments — true crime, history, personal anecdotes. |
| `technical_discussion` | 100 | In-depth technical or educational content. |
| `banter` | 75 | Casual conversation, jokes, co-host chatter. |
| `intro_outro` | 75 | Episode introductions and closings (NOT ads — things like "welcome to the show" or "thanks for listening"). |
| `news_recap` | 75 | Summarizing news stories or current events. |
| `product_mention_organic` | 100 | Genuine, unpaid discussion of a product or brand. Critical for avoiding false positives. |
| `self_promotion` | 75 | Host promoting their own Patreon, merch, book, tour, or side project. Sounds ad-like but is NOT a paid sponsorship. |
| `editorial_review` | 50 | Unpaid comparison or review of real products/brands. Evaluative language without sponsorship. |
| `url_mention_editorial` | 50 | Mentions URLs or links in an informational/editorial context (articles, papers, courses). |

---

## Generation Guidelines

### General Rules

1. **Vary podcast genres.** Distribute across: tech, true crime, comedy, sports, politics, health/wellness, business, science, pop culture, education.
2. **Vary host personas.** Some hosts are casual and conversational, others are formal and scripted. Mix male/female/non-binary names and speaking styles.
3. **Use realistic speech patterns.** Include filler words ("you know", "like", "um"), false starts, incomplete sentences, and natural disfluencies. Real transcripts from Whisper are messy.
4. **Vary segment length.** Segments should range from 3-10 sentences. Not every segment should be the same length.
5. **Do NOT include labels, markers, or metadata in the text itself.** The text field should read like a raw transcript with no indication of its classification.

### Ad Segment Rules

1. **Use realistic but fictional brand names.** Do NOT use real brands. Invent plausible names like "BrightPath VPN", "GroundUp Coffee", "FocusFrame Glasses", "NovaSleep Mattress".
2. **Include common ad patterns naturally:**
   - Promo codes: "use code PODCAST at checkout", "head to brightpath.com/showname"
   - Personal endorsements: "I've been using this for three months and honestly..."
   - Transition phrases: "this episode is brought to you by", "a quick word from our sponsor"
   - Call to action: "go check them out", "link in the show notes"
3. **Vary ad polish levels.** Some ads should sound scripted and professional. Others should sound like the host is reading from notes and ad-libbing. Some should sound like the host genuinely loves the product.
4. **Include edge cases:**
   - Ads that start mid-sentence after the host was discussing the topic
   - Ads that blend into the content ("speaking of sleep, let me tell you about NovaSleep...")
   - Very short ads (1-2 sentences like "quick shoutout to BrightPath VPN, link in the description")
   - Ads with no explicit transition phrase

### Non-Ad Segment Rules

1. **Include product mentions that are NOT ads.** This is critical for precision. Examples:
   - "I just switched to Firefox and honestly the performance is so much better"
   - "have you tried that new restaurant on 5th street"
   - "I've been reading this book called..."
2. **Include phrases that sound ad-like but aren't:**
   - "let me tell you about what happened" (sounds like ad transition but isn't)
   - "this is brought to you by years of hard work" (plays on ad phrasing)
   - "use your code to get into the building" (contains "use code" coincidentally)
3. **Include realistic intro/outro content** that discusses the show, Patreon, social media, or merch WITHOUT it being a paid sponsorship.
4. **Vary topics and tone.** Serious discussions, comedy bits, heated debates, calm explainers, etc.

---

## Timestamp & Episode Simulation Rules

Segments should be grouped into simulated podcast episodes to make timestamps realistic. This mirrors the real inference pipeline where Whisper transcribes an episode and produces timestamped chunks.

### Episode Structure

1. **Assign each segment to an episode.** Use episode IDs like `"ep_0001"`, `"ep_0002"`, etc.
2. **Each episode should contain 5-15 segments**, mixing ad and non-ad content. A typical episode structure:
   - 1-2 intro/banter segments at the start
   - A preroll ad near the beginning (optional)
   - Several content segments (interview, monologue, discussion, etc.)
   - 1-2 midroll ads roughly in the middle
   - More content segments
   - A possible late ad or cross-promo
   - 1 outro segment at the end
3. **Not every episode needs ads.** ~10% of episodes should have zero ad segments to reflect reality.

### Timestamp Rules

1. **Simulate total episode lengths of 20-90 minutes** (1,200 - 5,400 seconds).
2. **Segment durations should correlate with word count.** Use ~150 words per minute as the speaking rate. A 100-word segment is roughly 40 seconds, a 200-word segment is roughly 80 seconds.
3. **Timestamps must be sequential and non-overlapping within an episode.** Each segment's `start_time` must be >= the previous segment's `end_time`.
4. **Leave small gaps between segments** (0-5 seconds) to simulate natural pauses, music transitions, or un-transcribed filler.
5. **Ad segments should appear at realistic positions:**
   - Preroll ads: within the first 60-120 seconds
   - Midroll ads: between 30%-70% of total episode duration
   - Post-roll/outro ads: in the final 10% of the episode
6. **Round timestamps to 1 decimal place** (e.g., `847.2`, not `847.23456`).

### Episode Distribution

Across the full 2,000-segment dataset:
- Generate roughly **150-250 simulated episodes** (varying in length)
- Each episode should use a consistent podcast genre and host persona
- Distribute episodes evenly across the 10 podcast genres

---

## Quality Checks

After generation, verify the following:

1. **Balance**: Approximately 50/50 split between ad and non-ad.
2. **No label leakage**: The text alone should not contain metadata clues like "[AD]" or "label: 1".
3. **Diversity**: No more than 5% of segments should be in the same podcast genre with the same host name.
4. **Length distribution**: Segments should average ~100 words with a standard deviation of ~40 words. No segment should be under 20 words or over 300 words.
5. **Difficulty spread**: At least 10% of segments should be genuinely ambiguous or tricky (subtle ads, organic product mentions).
6. **Timestamp consistency**: Within each `episode_id`, segments should be in chronological order with no overlapping time ranges. `end_time` must be > `start_time` for every segment.
7. **Duration plausibility**: Segment durations (end - start) should roughly match word count at ~150 words/minute. A 100-word segment should be ~40s, not 5s or 300s.

---

## Example Segments

### Ad Example (host_read_ad)

```json
{"text": "So I want to take a second to talk about something I've actually been using a lot lately. BrightPath VPN. Look I know everyone and their mom has a VPN sponsor but I was genuinely surprised by how fast this one is. Like I was streaming on hotel wifi last week and it didn't even hiccup. If you want to check it out head to brightpath.com/techpod and you'll get three months free with the annual plan. Seriously worth it.", "label": 1, "category": "host_read_ad", "start_time": 847.2, "end_time": 903.6, "metadata": {"podcast_genre": "tech", "ad_style": "conversational", "episode_id": "ep_0042"}}
```

### Ad Example (subtle_ad)

```json
{"text": "you know what's been helping me stay on top of all this research is GroundUp Coffee. I started drinking it like two months ago because a friend recommended it and now I literally cannot go back to normal coffee. The focus is just different. Anyway they have a thing where if you go to groundup.co slash brainy you get twenty percent off your first order. Just throwing that out there.", "label": 1, "category": "subtle_ad", "start_time": 1423.0, "end_time": 1467.8, "metadata": {"podcast_genre": "science", "ad_style": "blended", "episode_id": "ep_0107"}}
```

### Non-Ad Example (product_mention_organic)

```json
{"text": "I actually just switched to a standing desk last month and oh my god the difference is insane. My back doesn't hurt anymore after long recording sessions. I got one of those ones from Ikea that you crank up manually because I didn't want to spend like eight hundred bucks on the motorized ones. Totally worth it though if you're sitting all day like we are.", "label": 0, "category": "product_mention_organic", "start_time": 312.5, "end_time": 351.9, "metadata": {"podcast_genre": "comedy", "episode_id": "ep_0015"}}
```

### Non-Ad Example (interview)

```json
{"text": "So when you first started researching this topic what was the thing that surprised you the most. Because I think for a lot of people they hear about deep sea mining and they just think oh yeah we need the minerals right. But there's this whole other side to it that nobody talks about. Can you walk us through what you found when you actually went down there.", "label": 0, "category": "interview", "start_time": 605.0, "end_time": 642.3, "metadata": {"podcast_genre": "science", "episode_id": "ep_0107"}}
```

---

## Batch Generation Strategy

To maintain variety, generate in batches. Each batch should produce **10-20 complete episodes** (~100 segments total). Each batch should:

1. Use a **different podcast genre** as the primary genre for that batch.
2. Use **2-3 different host personas** within the batch.
3. Contain a **mix of ad and non-ad categories** (not all one type).
4. Include **at least 5 deliberately tricky/ambiguous segments** per batch.
5. Generate segments **in episode order** — all segments for `ep_XXXX` should be produced together with sequential timestamps before moving to the next episode.

After all batches are complete, shuffle the entire dataset before saving. (Segments from the same episode will be scattered, but they can be reassembled by `episode_id` and sorted by `start_time` if needed.)

---

## File Output

Save the final dataset as:

```
project/data/synthetic_train.jsonl
```

One JSON object per line, UTF-8 encoded, no trailing newline.
