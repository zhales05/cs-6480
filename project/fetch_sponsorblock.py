"""
Fetch real-world sponsor segment data from the SponsorBlock API.

Strategy: The SponsorBlock API only supports per-video lookups (no global search).
This script takes a list of YouTube video IDs (from a file or auto-discovered via
yt-dlp channel search) and fetches sponsor segments for each.

Usage:
    # Auto-discover videos from podcast channels and fetch their sponsor segments:
    python project/fetch_sponsorblock.py --discover --limit 200

    # Or provide your own list of video IDs:
    python project/fetch_sponsorblock.py --video-ids-file my_ids.txt

    # Or pass IDs directly:
    python project/fetch_sponsorblock.py --video-ids "dQw4w9WgXcQ,abc123"
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

API_BASE = "https://sponsor.ajay.app"
SKIP_SEGMENTS_ENDPOINT = f"{API_BASE}/api/skipSegments"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "data", "sponsorblock_labels.jsonl")

# Popular podcast-style YouTube channels likely to have SponsorBlock data
PODCAST_CHANNELS = [
    "@lexfridman",
    "@veritasium",
    "@mkbhd",
    "@ColdFusion",
    "@TED",
    "@Fireship",
    "@NetworkChuck",
    "@ThePrimeTimeagen",
    "@TrashTaste",
    "@H3Podcast",
    "@ColinandSamir",
    "@MrBeast",
    "@LinusTechTips",
    "@SmarterEveryDay",
    "@WendoverProductions",
]

# Fallback: curated list of video IDs known to have sponsor segments
# (popular tech/podcast videos — these are real YouTube video IDs)
FALLBACK_VIDEO_IDS = []


def get_video_ids_from_channel(channel_handle, max_videos=20):
    """Use yt-dlp to get recent video IDs from a YouTube channel."""
    url = f"https://www.youtube.com/{channel_handle}/videos"
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "id",
        "--playlist-end", str(max_videos),
        "--quiet",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            ids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            return ids
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    Warning: yt-dlp failed for {channel_handle}: {e}")
    return []


def discover_video_ids(channels, videos_per_channel=15):
    """Discover video IDs from a list of YouTube channels."""
    all_ids = []
    for i, channel in enumerate(channels, 1):
        print(f"  [{i}/{len(channels)}] Scanning {channel}...", end=" ", flush=True)
        ids = get_video_ids_from_channel(channel, videos_per_channel)
        print(f"found {len(ids)} videos")
        all_ids.extend(ids)
    return list(dict.fromkeys(all_ids))  # deduplicate while preserving order


def fetch_sponsor_segments(video_id):
    """Fetch sponsor segments for a single video from SponsorBlock.

    Returns a list of segment dicts, or an empty list if none found.
    """
    params = {
        "videoID": video_id,
        "categories": json.dumps(["sponsor"]),
    }
    try:
        resp = requests.get(SKIP_SEGMENTS_ENDPOINT, params=params, timeout=10)
        if resp.status_code == 404:
            return []  # No segments for this video
        resp.raise_for_status()
        data = resp.json()
        segments = []
        for seg in data:
            segments.append({
                "start": round(seg["segment"][0], 1),
                "end": round(seg["segment"][1], 1),
                "votes": seg.get("votes", 0),
                "uuid": seg.get("UUID", ""),
                "category": seg.get("category", "sponsor"),
            })
        return segments
    except requests.RequestException:
        return []


def fetch_all(video_ids, min_votes=0, delay=0.3):
    """Fetch sponsor segments for all video IDs.

    Returns a list of dicts with video_id, segments, and segment_count.
    Only includes videos that have at least one qualifying segment.
    """
    results = []
    found = 0
    for i, vid in enumerate(video_ids, 1):
        print(f"  [{i}/{len(video_ids)}] {vid}...", end=" ", flush=True)
        segments = fetch_sponsor_segments(vid)

        # Filter by min votes
        segments = [s for s in segments if s["votes"] >= min_votes]

        if segments:
            results.append({
                "video_id": vid,
                "segments": segments,
                "segment_count": len(segments),
            })
            found += 1
            print(f"{len(segments)} sponsor segment(s)")
        else:
            print("no segments")

        if i < len(video_ids):
            time.sleep(delay)

    print(f"\n  Found sponsor segments in {found}/{len(video_ids)} videos")
    return results


def save_jsonl(records, output_path):
    """Write one JSON line per video to the output file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(records)} videos to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch sponsor segment labels from SponsorBlock for YouTube videos"
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Auto-discover video IDs from known podcast channels using yt-dlp"
    )
    parser.add_argument(
        "--video-ids", type=str, default=None,
        help="Comma-separated list of YouTube video IDs"
    )
    parser.add_argument(
        "--video-ids-file", type=str, default=None,
        help="Path to a text file with one video ID per line"
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Max total video IDs to process (default: 200)"
    )
    parser.add_argument(
        "--videos-per-channel", type=int, default=15,
        help="Videos to fetch per channel in discover mode (default: 15)"
    )
    parser.add_argument(
        "--min-votes", type=int, default=0,
        help="Minimum votes for a segment to be included (default: 0)"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output JSONL file path"
    )
    args = parser.parse_args()

    video_ids = []

    if args.video_ids:
        video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
        print(f"Using {len(video_ids)} provided video IDs")

    elif args.video_ids_file:
        with open(args.video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(video_ids)} video IDs from {args.video_ids_file}")

    elif args.discover:
        print(f"Discovering videos from {len(PODCAST_CHANNELS)} channels...")
        video_ids = discover_video_ids(PODCAST_CHANNELS, args.videos_per_channel)
        print(f"Discovered {len(video_ids)} unique video IDs\n")

    else:
        parser.error("Provide --discover, --video-ids, or --video-ids-file")

    if args.limit and len(video_ids) > args.limit:
        video_ids = video_ids[:args.limit]
        print(f"Limited to {args.limit} videos\n")

    print(f"Fetching sponsor segments (minVotes={args.min_votes})...")
    results = fetch_all(video_ids, min_votes=args.min_votes)

    if not results:
        print("No sponsor segments found.")
        return

    save_jsonl(results, args.output)
    total_segs = sum(r["segment_count"] for r in results)
    print(f"\nTotal: {len(results)} videos with {total_segs} sponsor segments")


if __name__ == "__main__":
    main()
