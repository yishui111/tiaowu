"""Synthesize a royalty-free demo dance track (120 BPM, 32s, 44.1kHz mono wav).

Used as offline test input for the dance generation models.
Usage: python make_demo_music.py out.wav
"""
import sys

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s
DURATION = 32.0
N = int(SR * DURATION)
t_all = np.arange(N) / SR


def env(n, a, d):
    """Simple attack-decay envelope."""
    e = np.ones(n)
    na, nd = int(a * SR), int(d * SR)
    if na > 0:
        e[:na] = np.linspace(0, 1, na)
    if nd > 0 and nd <= n:
        e[-nd:] *= np.linspace(1, 0, nd)
    return e


def kick(times):
    out = np.zeros(N)
    for tt in times:
        i0 = int(tt * SR)
        n = int(0.25 * SR)
        if i0 + n > N:
            break
        s = np.arange(n) / SR
        f = 120 * np.exp(-s * 22) + 42
        out[i0:i0 + n] += np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.002, 0.18)
    return out


def snare(times):
    out = np.zeros(N)
    rng = np.random.default_rng(7)
    for tt in times:
        i0 = int(tt * SR)
        n = int(0.2 * SR)
        if i0 + n > N:
            break
        s = np.arange(n) / SR
        noise = rng.standard_normal(n)
        body = np.sin(2 * np.pi * 190 * s)
        out[i0:i0 + n] += (0.7 * noise + 0.3 * body) * env(n, 0.001, 0.12)
    return out


def hat(times):
    out = np.zeros(N)
    rng = np.random.default_rng(11)
    for tt in times:
        i0 = int(tt * SR)
        n = int(0.06 * SR)
        if i0 + n > N:
            break
        noise = rng.standard_normal(n)
        # crude high-pass: first difference
        hp = np.diff(noise, prepend=0.0)
        out[i0:i0 + n] += 0.35 * hp * env(n, 0.001, 0.04)
    return out


def tone(freq, times, dur=0.22, gain=0.22, wave="saw"):
    out = np.zeros(N)
    for tt in times:
        i0 = int(tt * SR)
        n = int(dur * SR)
        if i0 + n > N:
            break
        s = np.arange(n) / SR
        if wave == "saw":
            x = 2 * ((freq * s) % 1.0) - 1
            x = 0.5 * x + 0.25 * np.sin(2 * np.pi * 2 * freq * s)
        else:
            x = np.sin(2 * np.pi * freq * s)
        out[i0:i0 + n] += gain * x * env(n, 0.01, 0.12)
    return out


beats = np.arange(0, DURATION, BEAT)
eighths = np.arange(0, DURATION, BEAT / 2)
kick_t = [b for i, b in enumerate(beats) if i % 2 == 0] + [beats[i] + BEAT * 0.75 for i in range(len(beats)) if i % 4 == 3]
snare_t = beats[1::2]
bass_notes = [55, 55, 65.4, 73.4]  # A1 A1 C2 D2
bass_t, idx = [], 0
for b in beats:
    bass_t.append(b)
    if idx % 2 == 1:
        bass_t.append(b + BEAT / 2)
    idx += 1
bass_freqs = [bass_notes[int(i / 2) % 4] for i in range(len(bass_t))]

audio = 1.0 * kick(sorted(kick_t)) + 0.55 * snare(snare_t) + 0.5 * hat(eighths)
for f, tt in zip(bass_freqs, bass_t):
    audio += tone(f, [tt], dur=0.42, gain=0.3)
chords = [(220.0, 261.6, 329.6), (196.0, 246.9, 293.7), (261.6, 329.6, 392.0), (174.6, 220.0, 261.6)]
for bar in range(int(DURATION / (BEAT * 4))):
    tt = bar * BEAT * 4
    chord = chords[bar % 4]
    for off in (0, BEAT * 2.5):
        for f in chord:
            audio += tone(f, [tt + off], dur=0.5, gain=0.08, wave="saw")

audio = audio / np.max(np.abs(audio)) * 0.9
fade = int(0.5 * SR)
audio[-fade:] *= np.linspace(1, 0, fade)
sf.write(sys.argv[1] if len(sys.argv) > 1 else "demo_music.wav", audio.astype(np.float32), SR)
print("wrote", sys.argv[1] if len(sys.argv) > 1 else "demo_music.wav", f"{DURATION}s @ {BPM} BPM")
