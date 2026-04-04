"""
Download YouTube audio and transcribe with Whisper, then label segments
using SponsorBlock timestamps to build a real-world test set.

Usage:
    python project/download_and_transcribe.py --input project/data/sponsorblock_labels.jsonl --limit 5
"""

import argparse
import json
import os
import subprocess
import tempfile

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "data", "sponsorblock_labels.jsonl")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "data", "real_test.jsonl")
DEFAULT_AUDIO_DIR = os.path.join(SCRIPT_DIR, "data", "audio")


def download_audio(video_id, output_dir):
    """Download audio from a YouTube video using yt-dlp.

    Returns the path to the downloaded audio file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "5",
        "--output", output_template,
        "--no-playlist",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] yt-dlp failed for {video_id}:")
        if e.stderr:
            for line in e.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [ERROR] Failed to download {video_id}: {e}")
        return None

    audio_path = os.path.join(output_dir, f"{video_id}.mp3")
    if os.path.exists(audio_path):
        return audio_path
    return None


def transcribe_audio(audio_path, model_name="base"):
    """Transcribe audio using Whisper.

    Returns a list of segment dicts with 'start', 'end', and 'text' keys.
    """
    try:
        import whisper
    except ImportError:
        print("  [ERROR] openai-whisper is not installed. Run: pip install openai-whisper")
        return None

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, verbose=False)
    return result.get("segments", [])


def merge_whisper_segments(whisper_segments, target_words=80):
    """Merge short Whisper chunks into larger segments of ~target_words words.

    Whisper produces 2-5 second chunks (~5-15 words each). This merges them
    into segments matching the training data format (~50-150 words).

    Returns a list of merged segment dicts with 'start', 'end', 'text'.
    """
    merged = []
    current_texts = []
    current_start = None
    current_end = None
    current_words = 0

    for seg in whisper_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        words = len(text.split())
        if current_start is None:
            current_start = seg["start"]

        current_texts.append(text)
        current_end = seg["end"]
        current_words += words

        if current_words >= target_words:
            merged.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_texts),
            })
            current_texts = []
            current_start = None
            current_end = None
            current_words = 0

    # flush any remaining text
    if current_texts and current_words >= 20:
        merged.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_texts),
        })

    return merged


def label_segments(whisper_segments, sponsor_segments, target_words=80):
    """Merge Whisper chunks into larger segments, then label as ad (1) or non-ad (0).

    A merged segment is labeled as ad if the majority of its duration overlaps
    with a SponsorBlock sponsor window.

    Args:
        whisper_segments: List of dicts from Whisper with 'start', 'end', 'text'.
        sponsor_segments: List of dicts with 'start' and 'end' from SponsorBlock.
        target_words: Approximate word count per output segment.

    Returns:
        List of labeled segment dicts.
    """
    merged = merge_whisper_segments(whisper_segments, target_words)

    labeled = []
    for seg in merged:
        start = seg["start"]
        end = seg["end"]
        duration = end - start

        # Measure overlap with sponsor windows
        overlap = 0.0
        for sponsor in sponsor_segments:
            overlap_start = max(start, sponsor["start"])
            overlap_end = min(end, sponsor["end"])
            if overlap_end > overlap_start:
                overlap += overlap_end - overlap_start

        # Label as ad if >20% of the segment overlaps with a sponsor window.
        # Lower threshold accounts for merged segments that straddle ad/non-ad boundaries.
        is_ad = 1 if duration > 0 and (overlap / duration) > 0.2 else 0

        labeled.append({
            "text": seg["text"],
            "label": is_ad,
            "start_time": round(start, 1),
            "end_time": round(end, 1),
        })

    return labeled


def process_video(video_id, sponsor_segments, audio_dir, whisper_model):
    """Download, transcribe, and label a single video.

    Returns a list of labeled segment dicts, or None on failure.
    """
    print(f"  Downloading audio...")
    audio_path = download_audio(video_id, audio_dir)
    if not audio_path:
        return None

    print(f"  Transcribing with Whisper ({whisper_model})...")
    whisper_segments = transcribe_audio(audio_path, whisper_model)
    if not whisper_segments:
        return None

    print(f"  Labeling {len(whisper_segments)} transcript segments...")
    labeled = label_segments(whisper_segments, sponsor_segments)

    ad_count = sum(1 for s in labeled if s["label"] == 1)
    print(f"  Result: {len(labeled)} segments ({ad_count} ad, {len(labeled) - ad_count} non-ad)")

    return labeled


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio, transcribe with Whisper, and label with SponsorBlock data"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="Input JSONL file from fetch_sponsorblock.py"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output JSONL file for labeled segments"
    )
    parser.add_argument(
        "--audio-dir", type=str, default=DEFAULT_AUDIO_DIR,
        help="Directory to store downloaded audio files"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of videos to process (default: all)"
    )
    parser.add_argument(
        "--whisper-model", type=str, default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--keep-audio", action="store_true",
        help="Keep downloaded audio files after processing"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        print("Run fetch_sponsorblock.py first to generate it.")
        return

    # Load SponsorBlock labels
    videos = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                videos.append(json.loads(line))

    if args.limit:
        videos = videos[:args.limit]

    # Load already-processed video IDs to skip them
    already_processed = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    vid = json.loads(line).get("video_id", "")
                    if vid:
                        already_processed.add(vid)
        if already_processed:
            print(f"Skipping {len(already_processed)} already-processed video(s)")

    videos = [v for v in videos if v["video_id"] not in already_processed]
    print(f"Processing {len(videos)} videos\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_segments = 0
    total_ads = 0
    processed = 0

    with open(args.output, "a") as out_f:
        for i, video in enumerate(videos, 1):
            video_id = video["video_id"]
            sponsor_segments = video["segments"]
            print(f"[{i}/{len(videos)}] Video: {video_id}")

            labeled = process_video(
                video_id, sponsor_segments, args.audio_dir, args.whisper_model
            )

            if labeled is None:
                print(f"  Skipped.\n")
                continue

            for seg in labeled:
                seg["video_id"] = video_id
                out_f.write(json.dumps(seg) + "\n")

            total_segments += len(labeled)
            total_ads += sum(1 for s in labeled if s["label"] == 1)
            processed += 1

            # Clean up audio if not keeping
            if not args.keep_audio:
                audio_path = os.path.join(args.audio_dir, f"{video_id}.mp3")
                if os.path.exists(audio_path):
                    os.remove(audio_path)

            print()

    print("=" * 50)
    print(f"Done! Processed {processed}/{len(videos)} videos")
    print(f"Total segments: {total_segments} ({total_ads} ad, {total_segments - total_ads} non-ad)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
