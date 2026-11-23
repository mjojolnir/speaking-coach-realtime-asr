# python poc/stream.py --model base.en
# python poc/stream.py --model small.en

# Text used to compare:
# Hello, my name is Jonathan and I am actually a computer science student. Basically, I am applying for this scholarship because I 
#  want to build impactful AI solutions. You know, speaking in English under pressure is literally quite challenging, so I am practicing my pacing today.

import argparse
import queue
import re
import sys
import time
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0  # Durasi tiap potongan audio untuk update real-time
LEXICAL_FILLERS = [
    "actually", "basically", "literally", "you know", "i mean", "sort of", "kind of"
]

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status] {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def count_fillers(text: str) -> dict:
    lower_text = text.lower()
    found = {}
    for filler in LEXICAL_FILLERS:
        matches = re.findall(rf"\b{re.escape(filler)}\b", lower_text)
        if matches:
            found[filler] = len(matches)
    return found


def main():
    parser = argparse.ArgumentParser(description="PoC Real-Time Local ASR Streaming")
    parser.add_argument("--model", type=str, default="base.en", choices=["base.en", "small.en"])
    parser.add_argument("--chunk", type=float, default=CHUNK_SECONDS, help="Chunk duration in seconds")
    args = parser.parse_args()

    print(f"Loading faster-whisper '{args.model}' on CUDA (float16)...")
    t_load = time.perf_counter()
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    print(f"Model ready in {time.perf_counter() - t_load:.2f}s.\n")
    print("🎙️  Mulai bicara dalam Bahasa Inggris (tekan Ctrl+C untuk berhenti)...\n")

    frames_per_chunk = int(SAMPLE_RATE * args.chunk)
    buffer = np.zeros((0, 1), dtype=np.float32)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                data = audio_queue.get()
                buffer = np.concatenate((buffer, data), axis=0)

                if len(buffer) >= frames_per_chunk:
                    chunk_audio = buffer[:frames_per_chunk].flatten()
                    buffer = buffer[frames_per_chunk:]

                    t0 = time.perf_counter()
                    segments, _ = model.transcribe(
                        chunk_audio,
                        language="en",
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    latency_ms = (time.perf_counter() - t0) * 1000

                    if text:
                        fillers = count_fillers(text)
                        filler_tag = f" | Fillers: {fillers}" if fillers else ""
                        print(f"[{latency_ms:5.0f} ms] {text}{filler_tag}")
                    else:
                        print(f"[{latency_ms:5.0f} ms] ... (silence / pause)")

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()