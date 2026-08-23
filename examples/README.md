# Example output

[`artemis-iii-training.md`](artemis-iii-training.md) is a real, unedited run of minute-ai: the
whole file exactly as the pipeline wrote it, summary and transcript included. Nothing was
hand-corrected, so it also shows the rough edges you should expect from a 3B model on CPU.

## Source

| | |
|---|---|
| Recording | *Artemis III Training*, [Houston We Have a Podcast](https://www.nasa.gov/podcasts/houston-we-have-a-podcast/artemis-iii-training/) (NASA), 21 August 2026 |
| Segment | 4 minutes, starting at 5:00 |
| Rights | Produced by NASA. NASA media is not subject to copyright in the United States and may be reused, including for derivative works such as transcripts. |

The audio itself is not committed: the repository stays free of large binaries, and `inputs/` is
git-ignored. Reproduce it with:

```bash
# 1. Take a four-minute segment from the episode
ffmpeg -ss 300 -t 240 -i <episode.mp3> -ac 1 -ar 16000 inputs/nasa-podcast-demo.wav

# 2. Run the pipeline
python main.py inputs/nasa-podcast-demo.wav \
    --model small \
    --language en \
    --speakers 2 \
    --summary-preset interview \
    --meeting-name "Artemis III Training"
```

## What it demonstrates

- **Speaker diarization**: two speakers, correctly separated and kept apart through the transcript.
- **The `interview` preset**: the summary is shaped as *Participants / Topics Covered / Key Points /
  Notable Quotes / Follow-ups* rather than the meeting default of decisions and action items.
- **Cleanup**: the transcript is the ASR output after the LLM pass, with punctuation and obvious
  mishearings repaired.

## Honest notes

Run on a CPU-only machine with `whisper-small` and `llama3.2:3b`, the smallest sensible pair. The
output is good but not flawless: the model expands "CTO" to a plausible guess rather than the term
actually used at NASA, and one quote in *Notable Quotes* runs two speakers' lines together. A larger
Whisper model and an 8B LLM tighten both, at the cost of speed.

Speaker labels stay as `SPEAKER_00` / `SPEAKER_01` because no `--speaker-names` were given.
