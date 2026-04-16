"""
Validates synthetic_train.jsonl against the spec in SYNTHETIC_DATA_GENERATION.md.
Usage: python project/validate_synthetic.py [path_to_jsonl]
Defaults to project/data/synthetic_train.jsonl
"""

import json
import sys
import re
from collections import Counter, defaultdict
from pathlib import Path

# Expected category counts from the spec
EXPECTED_AD_CATEGORIES = {
    "host_read_ad": 300,
    "promo_code_ad": 200,
    "preroll_ad": 100,
    "midroll_transition": 150,
    "product_testimonial_ad": 100,
    "cross_promo": 50,
    "subtle_ad": 100,
}

EXPECTED_NONAD_CATEGORIES = {
    "interview": 150,
    "monologue": 125,
    "storytelling": 125,
    "technical_discussion": 100,
    "banter": 75,
    "intro_outro": 75,
    "news_recap": 75,
    "product_mention_organic": 100,
    "self_promotion": 75,
    "editorial_review": 50,
    "url_mention_editorial": 50,
}

VALID_GENRES = {
    "tech", "true crime", "comedy", "sports", "politics",
    "health/wellness", "business", "science", "pop culture", "education",
}

VALID_AD_STYLES = {"conversational", "scripted", "blended"}

LEAKAGE_PATTERNS = [
    r"\[AD\]", r"\[ad\]", r"\[ADS?\]", r"\[SPONSOR\]",
    r"label:\s*[01]", r"\"label\"", r"\"category\"",
    r"\[NON.?AD\]", r"\[ADVERTISEMENT\]",
]


def load_data(path):
    segments = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                segments.append((i, obj))
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")
    return segments, errors


def check_required_fields(segments):
    errors = []
    required = {"text", "label", "category", "start_time", "end_time", "metadata"}
    for lineno, seg in segments:
        missing = required - set(seg.keys())
        if missing:
            errors.append(f"Line {lineno}: missing fields {missing}")
        if "metadata" in seg:
            if "podcast_genre" not in seg["metadata"]:
                errors.append(f"Line {lineno}: metadata missing 'podcast_genre'")
            if "episode_id" not in seg["metadata"]:
                errors.append(f"Line {lineno}: metadata missing 'episode_id'")
            if seg.get("label") == 1 and "ad_style" not in seg["metadata"]:
                errors.append(f"Line {lineno}: ad segment missing 'ad_style' in metadata")
    return errors


def check_label_balance(segments):
    labels = [seg["label"] for _, seg in segments]
    c = Counter(labels)
    total = len(labels)
    ratio = c.get(1, 0) / total if total else 0
    passed = 0.4 <= ratio <= 0.6
    detail = f"ads={c.get(1, 0)}, non-ads={c.get(0, 0)}, ratio={ratio:.2%}"
    return passed, detail


def check_category_distribution(segments):
    ad_cats = Counter()
    nonad_cats = Counter()
    for _, seg in segments:
        if seg["label"] == 1:
            ad_cats[seg["category"]] += 1
        else:
            nonad_cats[seg["category"]] += 1

    warnings = []

    for cat, expected in EXPECTED_AD_CATEGORIES.items():
        actual = ad_cats.get(cat, 0)
        if actual == 0:
            warnings.append(f"  ad '{cat}': expected ~{expected}, got 0")
        elif abs(actual - expected) / expected > 0.3:
            warnings.append(f"  ad '{cat}': expected ~{expected}, got {actual} (>{30}% off)")

    for cat, expected in EXPECTED_NONAD_CATEGORIES.items():
        actual = nonad_cats.get(cat, 0)
        if actual == 0:
            warnings.append(f"  non-ad '{cat}': expected ~{expected}, got 0")
        elif abs(actual - expected) / expected > 0.3:
            warnings.append(f"  non-ad '{cat}': expected ~{expected}, got {actual} (>{30}% off)")

    unknown_ad = set(ad_cats.keys()) - set(EXPECTED_AD_CATEGORIES.keys())
    unknown_nonad = set(nonad_cats.keys()) - set(EXPECTED_NONAD_CATEGORIES.keys())
    if unknown_ad:
        warnings.append(f"  unknown ad categories: {unknown_ad}")
    if unknown_nonad:
        warnings.append(f"  unknown non-ad categories: {unknown_nonad}")

    return len(warnings) == 0, warnings


def check_label_leakage(segments):
    issues = []
    compiled = [re.compile(p, re.IGNORECASE) for p in LEAKAGE_PATTERNS]
    for lineno, seg in segments:
        text = seg.get("text", "")
        for pattern in compiled:
            if pattern.search(text):
                issues.append(f"Line {lineno}: possible label leakage — matched '{pattern.pattern}'")
                break
    return len(issues) == 0, issues


def check_timestamps(segments):
    episodes = defaultdict(list)
    errors = []

    for lineno, seg in segments:
        start = seg.get("start_time")
        end = seg.get("end_time")
        ep_id = seg.get("metadata", {}).get("episode_id", "unknown")

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            errors.append(f"Line {lineno}: non-numeric timestamps")
            continue
        if end <= start:
            errors.append(f"Line {lineno}: end_time ({end}) <= start_time ({start})")
        if start < 0:
            errors.append(f"Line {lineno}: negative start_time ({start})")

        episodes[ep_id].append((lineno, start, end, seg))

    # Check sequential ordering within episodes
    for ep_id, segs in episodes.items():
        segs.sort(key=lambda x: x[1])  # sort by start_time
        for i in range(1, len(segs)):
            prev_end = segs[i - 1][2]
            curr_start = segs[i][1]
            if curr_start < prev_end:
                errors.append(
                    f"Episode {ep_id}: overlap — segment at line {segs[i][0]} "
                    f"starts at {curr_start} before previous ends at {prev_end}"
                )

    return len(errors) == 0, errors


def check_duration_plausibility(segments):
    warnings = []
    for lineno, seg in segments:
        start = seg.get("start_time", 0)
        end = seg.get("end_time", 0)
        duration = end - start
        word_count = len(seg.get("text", "").split())

        if word_count == 0:
            continue

        expected_duration = word_count / 150 * 60  # seconds
        ratio = duration / expected_duration if expected_duration > 0 else 0

        # Allow 0.3x to 3x tolerance
        if ratio < 0.3 or ratio > 3.0:
            warnings.append(
                f"Line {lineno}: {word_count} words, {duration:.1f}s duration "
                f"(expected ~{expected_duration:.1f}s, ratio={ratio:.2f})"
            )

    return len(warnings) == 0, warnings


def check_word_count_distribution(segments):
    word_counts = [len(seg.get("text", "").split()) for _, seg in segments]

    if not word_counts:
        return False, "No segments"

    import statistics
    avg = statistics.mean(word_counts)
    std = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
    under_20 = sum(1 for w in word_counts if w < 20)
    over_300 = sum(1 for w in word_counts if w > 300)

    details = [
        f"  avg={avg:.1f} words (target ~100)",
        f"  std={std:.1f} words (target ~40)",
        f"  under 20 words: {under_20}",
        f"  over 300 words: {over_300}",
        f"  min={min(word_counts)}, max={max(word_counts)}",
    ]

    passed = under_20 == 0 and over_300 == 0
    return passed, details


def check_genre_diversity(segments):
    combo_counts = Counter()
    total = len(segments)
    for _, seg in segments:
        genre = seg.get("metadata", {}).get("podcast_genre", "unknown")
        # We don't have host name in the required fields, so just check genre concentration
        combo_counts[genre] += 1

    warnings = []
    for genre, count in combo_counts.most_common():
        pct = count / total * 100
        if pct > 15:  # each of 10 genres should be ~10%, flag if >15%
            warnings.append(f"  genre '{genre}': {count} segments ({pct:.1f}%)")

    return len(warnings) == 0, warnings


def check_valid_values(segments):
    errors = []
    for lineno, seg in segments:
        label = seg.get("label")
        if label not in (0, 1):
            errors.append(f"Line {lineno}: invalid label '{label}'")

        genre = seg.get("metadata", {}).get("podcast_genre", "")
        if genre and genre not in VALID_GENRES:
            errors.append(f"Line {lineno}: unknown genre '{genre}'")

        if seg.get("label") == 1:
            style = seg.get("metadata", {}).get("ad_style", "")
            if style and style not in VALID_AD_STYLES:
                errors.append(f"Line {lineno}: unknown ad_style '{style}'")
    return len(errors) == 0, errors


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "project/data/synthetic_train.jsonl"
    path = Path(path)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Validating: {path}\n")

    segments, parse_errors = load_data(path)
    results = []

    # 1. JSON parsing
    if parse_errors:
        results.append(("JSON Parsing", False, parse_errors[:10]))
    else:
        results.append(("JSON Parsing", True, [f"{len(segments)} segments parsed"]))

    if not segments:
        print("No valid segments found. Aborting.")
        sys.exit(1)

    # 2. Required fields
    field_errors = check_required_fields(segments)
    results.append(("Required Fields", len(field_errors) == 0, field_errors[:10]))

    # 3. Valid values
    passed, errors = check_valid_values(segments)
    results.append(("Valid Values", passed, errors[:10]))

    # 4. Label balance
    passed, detail = check_label_balance(segments)
    results.append(("Label Balance", passed, [detail]))

    # 5. Category distribution
    passed, warnings = check_category_distribution(segments)
    results.append(("Category Distribution", passed, warnings))

    # 6. Label leakage
    passed, issues = check_label_leakage(segments)
    results.append(("No Label Leakage", passed, issues[:10]))

    # 7. Timestamps
    passed, errors = check_timestamps(segments)
    results.append(("Timestamp Consistency", passed, errors[:10]))

    # 8. Duration plausibility
    passed, warnings = check_duration_plausibility(segments)
    results.append(("Duration Plausibility", passed, warnings[:10]))

    # 9. Word count distribution
    passed, details = check_word_count_distribution(segments)
    results.append(("Word Count Distribution", passed, details))

    # 10. Genre diversity
    passed, warnings = check_genre_diversity(segments)
    results.append(("Genre Diversity", passed, warnings))

    # Print results
    print("=" * 60)
    all_passed = True
    for name, passed, details in results:
        status = "PASS" if passed else "WARN"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}")
        for d in details:
            print(f"       {d}")
    print("=" * 60)

    total = len(segments)
    ads = sum(1 for _, s in segments if s["label"] == 1)
    episodes = len(set(s.get("metadata", {}).get("episode_id", "") for _, s in segments))
    print(f"\nSummary: {total} segments, {ads} ads, {total - ads} non-ads, {episodes} episodes")

    if all_passed:
        print("\nAll checks passed!")
    else:
        print("\nSome checks have warnings — review above.")


if __name__ == "__main__":
    main()
