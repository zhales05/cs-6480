# Project Proposal: Automatic Advertisement Detection in Podcasts

*Deep Learning Course Project Proposal*

---

## Overview

The goal of this project is to build a model that can automatically detect advertisement segments in podcast audio. This has obvious real-world utility — anyone who has sat through a mid-roll ad knows the pain. The project will use a combination of synthetic data generation and real-world labeled data to train a text-based classifier, with a stretch goal of adding an audio-based CNN classifier for comparison.

## Problem Statement

Given a podcast episode, identify the start and end timestamps of any sponsored or advertisement segments. This is framed as a binary classification problem: ad content vs. non-ad content. Applications include ad-skipping tools, content indexing, and podcast analytics.

## Data Collection

Data will come from two sources:

**Synthetic data via LLM:** An LLM will be used to generate realistic ad transcript segments (e.g. host-read sponsorships, promo codes, brand callouts) as well as normal podcast dialogue. This bootstraps training data quickly without requiring manual labeling. The tradeoff is that synthetic text tends to be cleaner and more uniform than real transcripts, so generalization to messy real-world audio may be limited.

**SponsorBlock + yt-dlp:** SponsorBlock is an open-source, crowdsourced database of sponsor segment timestamps across YouTube videos, with millions of submissions. Using their public API at sponsor.ajay.app, video IDs and labeled timestamps can be pulled for free. Audio will then be downloaded with yt-dlp and sliced into labeled segments. This real-world data will serve primarily as the validation and test set, allowing us to measure how well the model trained on synthetic data generalizes.

## Approach

**Primary Objective — Text-Based Classifier:** Audio will first be transcribed using OpenAI's Whisper model. The resulting transcript will be split into segments and classified as ad or non-ad using a fine-tuned BERT model, with a TF-IDF + logistic regression baseline for comparison. Training will use LLM-generated synthetic data, with evaluation on real SponsorBlock examples.

**Stretch Goal — CNN Audio Classifier:** If time permits, a second approach will be explored using raw audio. Each labeled audio segment will be converted into a mel spectrogram — a visual representation of audio frequency over time — and a CNN will be trained to classify these spectrogram images as ad or non-ad. This approach connects directly to CNN coursework and would allow for a meaningful comparison between the text and audio methods.

## Baseline

A simple keyword and regex matcher will serve as the baseline. Phrases like "sponsored by", "promo code", "use code", and "this episode is brought to you by" are highly predictive of ad content and provide a strong, interpretable lower bound to beat.

## Evaluation

Models will be evaluated using precision, recall, and F1-score on an 80/20 train/test split. Recall will be weighted more heavily in analysis — missing an ad segment is a less severe error than incorrectly flagging real content. A key discussion point will be the performance gap between synthetic training data and real-world test data, which will shed light on the limitations of LLM-generated training sets.

## Deliverables

The primary deliverable is a trained text classifier with evaluation results comparing it to the keyword baseline. A simple demo will be built that accepts a podcast episode and outputs timestamped ad segments. If the stretch goal is completed, a comparative analysis between the text classifier and the CNN audio approach will also be included.

## Why This Project

This project is compelling because it solves a real, relatable problem with a clean and well-scoped approach. The data pipeline using SponsorBlock and yt-dlp is novel and automated, reducing the manual labeling burden significantly. The use of synthetic training data from an LLM is an interesting methodological choice with honest tradeoffs worth analyzing. And if the CNN stretch goal is pursued, the project naturally connects classroom CNN theory to a practical audio application.
