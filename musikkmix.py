#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
musikkmix – lager en beat-synkronisert musikkvideo av loop-videoer + en mp3.

Installasjon (Ubuntu):
    sudo apt install ffmpeg python3-venv
    python3 -m venv ~/.venvs/musikkmix && source ~/.venvs/musikkmix/bin/activate
    pip install numpy scipy scikit-learn librosa opencv-python-headless
    # valgfritt (bedre visuell «forståelse» + tema-søk):
    pip install torch transformers pillow

Eksempel:
    ./musikkmix.py musikk.mp3 klipp/*.mp4 -o mix.mp4
    ./musikkmix.py musikk.mp3 klipp/ -o mix.mp4 --resolution 3840x2160 --intensity 1.3 --clip --theme "neon city at night"
    ./musikkmix.py musikk.mp3 klipp/ --limit 40 --plan-only      # raskt: bare lag klippeplan (JSON)

Hvordan det virker:
  1. LYD: tempo, beats, downbeats (taktslag), energi per beat (RMS + onset + lysstyrke),
     struktur (agglomerativ segmentering av MFCC/chroma), gruppering av like deler (KMeans),
     breaks (energidipp), drops (plutselig energiøkning) og build-ups (opptakt før drop).
  2. VIDEO: bevegelse over tid, lysstyrke, metning, fargeprofil og (valgfritt) CLIP-embeddings.
  3. MATCHING: hver musikkdel får en «pool» av videoer som passer energien og ser forskjellige ut
     fra andre deler. Like deler (f.eks. refreng) får samme look. Innen en del velges video
     og startpunkt etter energi, variasjon og visuell kontinuitet.
  4. KLIPPING: klippene lengdes etter energi (høy energi = 1–2 beats, rolig = 8–16 beats),
     og alltid på beat/takt. Drops får hard treff, før drop korter klippene seg (build-up).
  5. EFFEKTER (valgfritt): crossfade i rolige partier, zoom-puls på beat/takt/kick,
     shake på kick og lysere/mørkere pulsering på kick. Se gruppen «Effekter» i --help.
  6. RENDER: ffmpeg, frame-nøyaktig, og mp3 legges på.

Eksempel med effekter:
    ./musikkmix.py musikk.mp3 klipp/ --fx-preset punchy
    ./musikkmix.py musikk.mp3 klipp/ --crossfade 0.2 --zoom 0.05 --zoom-on beat --pulse 0.12 --pulse-on kick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

try:
    import numpy as np
    import cv2
    import librosa
    from scipy.ndimage import uniform_filter1d, maximum_filter1d
    from scipy.signal import butter, sosfilt
    from sklearn.cluster import KMeans
except ImportError as e:  # pragma: no cover
    sys.exit(f"Mangler Python-pakke: {e.name}\nSe installasjonsinstruksjonene øverst i filen.")

LENGTHS = [1, 2, 4, 8, 16]  # mulige klipplengder i beats


def log(*a):
    print(*a, flush=True)


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Kommando feilet: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


# --------------------------------------------------------------------------------------
# Hjelpere
# --------------------------------------------------------------------------------------
def rnorm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.percentile(x, [5, 95])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def mask_runs(mask, minlen):
    runs, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= minlen:
                runs.append((start, i))
            start = None
    return runs


def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=duration:format=duration", "-of", "json", path])
    j = json.loads(out)
    for src in (j["streams"][0].get("duration") if j.get("streams") else None,
                j.get("format", {}).get("duration")):
        try:
            return float(src)
        except (TypeError, ValueError):
            pass
    raise RuntimeError(f"Fant ikke varighet for {path}")


# --------------------------------------------------------------------------------------
# LYD-ANALYSE
# --------------------------------------------------------------------------------------
@dataclass
class Audio:
    duration: float
    tempo: float
    bt: np.ndarray                 # tidspunkt for intervallgrenser (len = nb+1)
    energy: np.ndarray             # per beat-intervall, 0..1
    phase: int                     # hvilken beat (mod 4) som er «1» i takten
    sections: list = field(default_factory=list)   # [(a,b)] i beat-indekser
    clusters: list = field(default_factory=list)   # klyngeid per seksjon
    breaks: list = field(default_factory=list)     # [(a,b)]
    drops: list = field(default_factory=list)      # beat-indekser
    buildup: np.ndarray = None
    kicks: np.ndarray = None           # tidspunkt (s) for kick/bassanslag
    kick_strength: np.ndarray = None   # 0..1


def snap_downbeat(b, phase, nb):
    r = (b - phase) % 4
    d = -r if r <= 2 else 4 - r
    return int(np.clip(b + d, 0, nb - 1))


def analyze_audio(path, workdir, section_seconds, sr=22050, hop=512):
    wav = os.path.join(workdir, "audio.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), wav])
    y, _ = librosa.load(wav, sr=sr, mono=True)
    dur = len(y) / sr
    log(f"  Lydlengde: {dur:.1f} s")

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    nfr = min(len(onset), len(rms), len(cent), mfcc.shape[1], chroma.shape[1])
    onset, rms, cent, mfcc, chroma = onset[:nfr], rms[:nfr], cent[:nfr], mfcc[:, :nfr], chroma[:, :nfr]

    tempo, bf = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    bf = bf[bf < nfr]
    if len(bf) < 16:
        raise RuntimeError("Fant for få beats i lydfilen – er musikken rytmisk nok?")
    fixed = librosa.util.fix_frames(bf, x_min=0, x_max=nfr)
    bt = librosa.frames_to_time(fixed, sr=sr, hop_length=hop)
    bt[-1] = dur
    nb = len(fixed) - 1
    log(f"  Tempo ≈ {tempo:.1f} BPM, {nb} beat-intervaller")

    def sync(x):
        return librosa.util.sync(x, fixed, aggregate=np.mean)

    onset_b = sync(onset[None])[0]
    db_b = librosa.amplitude_to_db(sync(rms[None])[0] + 1e-6, ref=np.max)
    cent_b = sync(cent[None])[0]
    energy = 0.55 * rnorm(db_b) + 0.30 * rnorm(onset_b) + 0.15 * rnorm(cent_b)
    energy = rnorm(uniform_filter1d(energy, 3, mode="nearest"))

    # downbeat-fase: hvilken av 4 posisjoner har sterkest onset i snitt
    idx = np.arange(nb)
    scores = [onset_b[(idx % 4 == p) & (idx >= 1)].mean() for p in range(4)]
    phase = int(np.argmax(scores))

    # kicks: plutselige økninger i lavfrekvent energi (< 150 Hz)
    sos = butter(4, 150, btype="low", fs=sr, output="sos")
    lrms = librosa.feature.rms(y=sosfilt(sos, y), frame_length=1024, hop_length=hop)[0]
    ldb = librosa.amplitude_to_db(lrms + 1e-6, ref=np.max)
    lenv = uniform_filter1d(np.maximum(0.0, np.diff(ldb, prepend=ldb[0])), 2, mode="nearest")
    kpeaks = librosa.util.peak_pick(lenv, pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                    delta=0.5 * float(lenv.std()), wait=max(1, int(0.12 * sr / hop)))
    kicks = librosa.frames_to_time(kpeaks, sr=sr, hop_length=hop)
    if len(kpeaks):
        ks = lenv[kpeaks]
        kick_strength = np.clip(ks / (np.percentile(ks, 90) + 1e-9), 0, 1)
    else:
        kick_strength = np.zeros(0)

    # seksjoner
    feat = np.vstack([sync(mfcc), sync(chroma)])
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-9)
    k = int(np.clip(round(dur / section_seconds), 2, max(2, nb // 8)))
    bounds = librosa.segment.agglomerative(feat, k)
    snapped = [0]
    for b in sorted(set(int(x) for x in bounds)):
        if b == 0:
            continue
        s = snap_downbeat(b, phase, nb)
        if s - snapped[-1] >= 8 and nb - s >= 8:
            snapped.append(s)
    sections = [(snapped[i], snapped[i + 1] if i + 1 < len(snapped) else nb) for i in range(len(snapped))]

    clusters = [0] * len(sections)
    if len(sections) >= 2:
        sf = np.array([feat[:, a:b].mean(axis=1) for a, b in sections])
        nc = min(len(sections), max(2, int(round(len(sections) / 2))))
        clusters = [int(c) for c in KMeans(n_clusters=nc, n_init=10, random_state=0).fit_predict(sf)]

    # breaks: energi faller kraftig i forhold til omgivelsene
    e_s = uniform_filter1d(energy, 4, mode="nearest")
    ref = maximum_filter1d(e_s, size=33, mode="nearest")
    breaks = mask_runs((e_s < 0.45 * ref) & (ref > 0.55), 4)

    # drops: plutselig økning i energi (snitt av neste 4 vs forrige 4 beats)
    jump = np.zeros(nb)
    for i in range(4, nb - 3):
        jump[i] = energy[i:i + 4].mean() - energy[i - 4:i].mean()
    peaks = (jump > 0.28) & (jump >= maximum_filter1d(jump, 17, mode="nearest"))
    drops = []
    for i in np.where(peaks)[0]:
        s = snap_downbeat(int(i), phase, nb)
        if not drops or s - drops[-1] >= 16:
            drops.append(s)

    buildup = np.zeros(nb, dtype=bool)
    for d in drops:
        buildup[max(0, d - 8):d] = True

    log(f"  {len(sections)} seksjoner, {len(set(clusters))} grupper, {len(breaks)} breaks, {len(drops)} drops, {len(kicks)} kicks")
    return Audio(dur, tempo, bt, energy, phase, sections, clusters, breaks, drops, buildup, kicks, kick_strength)


# --------------------------------------------------------------------------------------
# VIDEO-ANALYSE
# --------------------------------------------------------------------------------------
@dataclass
class Video:
    path: str
    name: str
    duration: float
    sample_fps: float
    motion: np.ndarray
    brightness: float
    saturation: float
    desc: np.ndarray
    frames: list = field(default_factory=list)
    motion_n: np.ndarray = None
    energy: float = 0.0
    emb: np.ndarray = None


def analyze_video(path, want_frames=0, sample_fps=4.0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Kan ikke åpne {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / sample_fps)))
    est = max(1, nframes // step)
    keep_every = max(1, est // want_frames) if want_frames else 0
    prev, motion, sats, vals, frames = None, [], [], [], []
    hue = np.zeros(12)
    idx = k = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(fr, (96, 54), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            motion.append(float(np.mean(np.abs(gray - prev))) / 255 if prev is not None else 0.0)
            prev = gray
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            s = hsv[..., 1].ravel().astype(np.float32) / 255
            hue += np.histogram(hsv[..., 0].ravel(), bins=12, range=(0, 180), weights=s)[0]
            sats.append(float(s.mean()))
            vals.append(float(hsv[..., 2].mean()) / 255)
            if keep_every and k % keep_every == 0:
                frames.append(cv2.cvtColor(cv2.resize(fr, (224, 224)), cv2.COLOR_BGR2RGB))
            k += 1
        idx += 1
    cap.release()
    if len(motion) < 2:
        raise RuntimeError(f"For få bilder i {path}")
    motion[0] = motion[1]
    hue = hue / (hue.sum() + 1e-9)
    desc = np.concatenate([hue, [np.mean(sats), np.mean(vals)]])
    dur = probe_duration(path)
    return Video(path, os.path.basename(path), dur, fps / step, np.array(motion),
                 float(np.mean(vals)), float(np.mean(sats)), desc, frames[:16])


def normalize_videos(vids):
    allm = np.concatenate([uniform_filter1d(v.motion, 3, mode="nearest") for v in vids])
    lo, hi = np.percentile(allm, [5, 95])
    hi = max(hi, lo + 1e-6)
    for v in vids:
        v.motion_n = np.clip((uniform_filter1d(v.motion, 3, mode="nearest") - lo) / (hi - lo), 0, 1)
    means = np.array([v.motion_n.mean() for v in vids])
    if len(vids) >= 2 and means.max() - means.min() > 0.05:
        a, b = means.min(), means.max()
        for v in vids:
            v.motion_n = np.clip((v.motion_n - a) / (b - a), 0, 1)
    for v in vids:
        v.energy = float(v.motion_n.mean())


class ClipEmbedder:
    def __init__(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise RuntimeError("--clip krever: pip install torch transformers pillow")
        self.torch = torch
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def _tensor(self, out):
        return out if self.torch.is_tensor(out) else out.pooler_output

    def images(self, frames):
        with self.torch.no_grad():
            f = self._tensor(self.model.get_image_features(**self.proc(images=frames, return_tensors="pt")))
        f = f.mean(0).numpy()
        return f / (np.linalg.norm(f) + 1e-9)

    def text(self, t):
        with self.torch.no_grad():
            f = self._tensor(self.model.get_text_features(**self.proc(text=[t], return_tensors="pt", padding=True)))
        f = f[0].numpy()
        return f / (np.linalg.norm(f) + 1e-9)


def similarity_matrix(vids):
    if all(v.emb is not None for v in vids):
        X = np.array([v.emb for v in vids])
    else:
        X = np.array([v.desc / (np.linalg.norm(v.desc) + 1e-9) for v in vids])
    S = X @ X.T
    if len(vids) > 2:
        off = S[~np.eye(len(vids), dtype=bool)]
        S = np.clip((S - off.mean()) / (off.std() + 1e-6), -2, 2) / 2
    return S


# --------------------------------------------------------------------------------------
# KLIPPEPLAN
# --------------------------------------------------------------------------------------
@dataclass
class Slot:
    i0: int
    i1: int
    t0: float
    t1: float
    energy: float
    target: float
    section: int
    cluster: int
    kind: str
    video: str = ""
    video_idx: int = -1
    offset: float = 0.0


def plan_cuts(A: Audio, a, rng):
    nb = len(A.energy)
    sec_of = np.zeros(nb, int)
    for si, (s0, s1) in enumerate(A.sections):
        sec_of[s0:s1] = si
    sec_starts = [s0 for s0, _ in A.sections] + [nb]
    brk = np.zeros(nb, bool)
    for x, y in A.breaks:
        brk[x:y] = True
    drops = sorted(A.drops)
    dur_beat = np.diff(A.bt)
    min_idx = LENGTHS.index(min(LENGTHS, key=lambda L: abs(L - a.min_cut_beats)))
    max_idx = LENGTHS.index(min(LENGTHS, key=lambda L: abs(L - a.max_cut_beats)))
    slots, i = [], 0
    while i < nb:
        e = float(A.energy[i])
        ea = min(1.0, e * a.intensity)
        idx = 0 if ea > 0.85 else 1 if ea > 0.65 else 2 if ea > 0.4 else 3 if ea > 0.2 else 4
        if rng.random() < 0.25:
            idx += rng.choice([-1, 1])
        kind = "normal"
        nxt_drop = next((d for d in drops if d > i), None)
        if i in drops:
            kind, idx = "drop", 1
        elif brk[i]:
            kind, idx = "break", max(idx, 3)
        elif A.buildup[i] and nxt_drop is not None:
            kind = "buildup"
            dist = nxt_drop - i
            idx = 0 if dist <= 2 else min(idx, 1) if dist <= 4 else max(0, idx - 1)
        idx = int(np.clip(idx, min_idx, max_idx))
        while idx < 4 and dur_beat[i] * LENGTHS[idx] < a.min_cut_seconds:
            idx += 1
        L = LENGTHS[idx]
        off = (i - A.phase) % 16
        L = min(L, 16 if off == 0 else (off & -off))
        L = min(L, next(s for s in sec_starts if s > i) - i, nb - i)
        if nxt_drop is not None and i < nxt_drop < i + L:
            L = nxt_drop - i
        L = max(1, L)
        e_mean = float(A.energy[i:i + L].mean())
        target = 1.0 if kind == "drop" else min(e_mean, 0.35) if kind == "break" else e_mean
        sec = int(sec_of[i])
        slots.append(Slot(i, i + L, float(A.bt[i]), float(A.bt[i + L]), e_mean, target,
                          sec, A.clusters[sec], kind))
        i += L
    return slots


def assign_videos(slots, A, vids, sim, a, rng, theme):
    nv = len(vids)
    nclusters = len(set(A.clusters)) or 1
    cl_energy = {}
    for si, c in enumerate(A.clusters):
        s0, s1 = A.sections[si]
        cl_energy.setdefault(c, []).append(float(A.energy[s0:s1].mean()))
    cl_energy = {c: float(np.mean(v)) for c, v in cl_energy.items()}
    pool_size = a.pool_size or int(np.clip(math.ceil(nv / nclusters) + 1, 2, nv))
    pool_size = min(pool_size, nv)
    claimed = np.zeros(nv)
    pools = {}

    def make_pool(c):
        ce = cl_energy.get(c, 0.5)
        sc = np.array([-1.5 * abs(v.energy - ce) - 0.5 * claimed[j] + 0.3 * theme[j] + 0.3 * rng.random()
                       for j, v in enumerate(vids)])
        top = list(np.argsort(-sc)[:pool_size])
        claimed[top] += 1
        return set(int(t) for t in top)

    usage = np.zeros(nv)
    used_offsets = {j: [] for j in range(nv)}
    recent, last = [], -1
    T = max(0.05, a.randomness)
    for s in slots:
        if s.cluster not in pools:
            pools[s.cluster] = make_pool(s.cluster)
        pool = pools[s.cluster]
        sc = np.zeros(nv)
        for j, v in enumerate(vids):
            x = 2.0 * (1 - abs(v.energy - s.target))
            x += 0.8 if j in pool else 0.0
            x += 0.4 * theme[j]
            if s.kind == "drop":
                x += 0.8 * v.energy
            if last >= 0:
                x += a.continuity * sim[last, j]
            for d, r in enumerate(reversed(recent[-4:]), 1):
                if r == j:
                    x -= 1.2 / d
            if nv > 1 and j == last:
                x -= 3.0
            x -= 0.3 * (usage[j] - usage.mean()) / (1 + usage.mean())
            sc[j] = x
        p = np.exp((sc - sc.max()) / T)
        p /= p.sum()
        j = int(rng.choices(range(nv), weights=list(p))[0])
        v = vids[j]
        s.video, s.video_idx = v.name, j
        s.offset = choose_offset(v, s.t1 - s.t0, s.target, used_offsets[j], rng)
        used_offsets[j].append(s.offset)
        usage[j] += 1
        recent.append(j)
        last = j


def choose_offset(v, dur, target, used, rng, step=0.5):
    cands = np.arange(0, v.duration, step) if v.duration > step else np.array([0.0])
    n_s = max(1, int(round(dur * v.sample_fps)))
    L = len(v.motion_n)
    best, best_cost = 0.0, 1e9
    for o in cands:
        i0 = int(o * v.sample_fps)
        m = v.motion_n[(i0 + np.arange(n_s)) % L].mean()
        cost = abs(m - target) + 0.1 * rng.random()
        for u in used:
            d = abs(o - u)
            d = min(d, v.duration - d)
            if d < 1.5:
                cost += 0.25 * (1 - d / 1.5)
        if cost < best_cost:
            best, best_cost = float(o), cost
    return best


# --------------------------------------------------------------------------------------
# RENDER
# --------------------------------------------------------------------------------------
def build_cache(v, need, W, H, fps, workdir, a):
    out = os.path.join(workdir, f"cache_{abs(hash(v.path)) % 10**8}.mp4")
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={fps},setsar=1,format=yuv420p")
    run(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", v.path, "-t", f"{need:.3f}", "-an",
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-g", "30", out])
    return out


ZOOM_DECAY, SHAKE_DECAY, PULSE_DECAY = 7.0, 14.0, 10.0


def triggers(A, s, mode, a, f0, fps):
    """Tidspunkt (i bilder relativt til klippstart) og styrke for effekt-utløsere i et klipp."""
    raw = []
    if mode in ("beat", "bar"):
        for j in range(max(1, s.i0), s.i1):
            if mode == "bar" and (j - A.phase) % 4:
                continue
            raw.append((float(A.bt[j]), 1.0))
    elif mode == "kick" and A.kicks is not None and len(A.kicks):
        m = (A.kicks >= s.t0 - 0.02) & (A.kicks < s.t1)
        raw = [(float(t), 0.5 + 0.5 * float(w)) for t, w in zip(A.kicks[m], A.kick_strength[m])]
    return [(max(0.0, (t + a.audio_offset) * fps - f0), w) for t, w in raw]


def env_expr(trig, decay, fps):
    """ffmpeg-uttrykk: sum av eksponentielt avtagende pulser (n = bildenummer i klippet)."""
    if not trig:
        return "0"
    terms = [f"{w:.2f}*gte(n,{f:.2f})*exp(-{decay:.1f}*max(0,n-{f:.2f})/{fps})" for f, w in trig]
    return "min(1," + "+".join(terms) + ")"


def effect_chain(s, A, a, W, H, fps, f0):
    parts = []
    boost = 1.6 if s.kind == "drop" else 1.0
    if a.zoom > 0 or a.shake > 0:
        zt = triggers(A, s, a.zoom_on, a, f0, fps) if a.zoom > 0 else []
        st = triggers(A, s, a.shake_on, a, f0, fps) if a.shake > 0 else []
        z = f"({1 + 2 * a.shake:.4f}+{a.zoom * boost:.4f}*{env_expr(zt, ZOOM_DECAY, fps)})"
        parts.append(f"scale=w='trunc(iw*{z}/2)*2':h='trunc(ih*{z}/2)*2':eval=frame")
        env = env_expr(st, SHAKE_DECAY, fps)
        ax, ay = a.shake * W, a.shake * H
        parts.append(f"crop={W}:{H}"
                     f":x='max(0,min(iw-ow,(iw-ow)/2+{ax:.2f}*{env}*sin(n*13.7)))'"
                     f":y='max(0,min(ih-oh,(ih-oh)/2+{ay:.2f}*{env}*sin(n*9.3)))'")
    if a.pulse > 0:
        pt = triggers(A, s, a.pulse_on, a, f0, fps)
        sign = 1 if a.pulse_mode == "light" else -1
        parts.append(f"eq=brightness='{sign * a.pulse * boost:.3f}*{env_expr(pt, PULSE_DECAY, fps)}':eval=frame")
    if a.drop_flash and s.kind == "drop":
        parts.append("fade=t=in:st=0:d=0.25:color=white")
    return parts


def render(slots, vids, A, audio_path, out_path, a, workdir, end_time):
    W, H = (int(x) for x in a.resolution.lower().split("x"))
    W, H = W - W % 2, H - H % 2
    fps = a.fps

    need = {}
    for s in slots:
        need[s.video_idx] = max(need.get(s.video_idx, 0), vids[s.video_idx].duration + (s.t1 - s.t0) + 1.5)
    log(f"Normaliserer {len(need)} videoer til {W}x{H} @ {fps} fps …")
    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        caches = dict(zip(need, ex.map(lambda j: build_cache(vids[j], need[j], W, H, fps, workdir, a), need)))

    entries = []
    for k, s in enumerate(slots):
        t0 = 0.0 if k == 0 else s.t0 + a.audio_offset
        t1 = end_time if k == len(slots) - 1 else s.t1 + a.audio_offset
        f0, f1 = round(t0 * fps), round(t1 * fps)
        if f1 - f0 >= 1:
            entries.append({"s": s, "f0": f0, "nf": f1 - f0})

    jobs, clips = [], []
    n_xf = 0
    for k, e in enumerate(entries):
        s, f0, nf = e["s"], e["f0"], e["nf"]
        prev = entries[k - 1] if k > 0 else None
        chain = effect_chain(s, A, a, W, H, fps, f0)
        xf = 0
        if prev and a.crossfade > 0 and s.kind != "drop" and s.energy <= a.crossfade_max_energy:
            xf = min(int(round(a.crossfade * (2 if s.kind == "break" else 1) * fps)), nf // 2)
            if xf < 2:
                xf = 0
        clip = os.path.join(workdir, f"clip_{k:05d}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{s.offset:.3f}", "-i", caches[s.video_idx]]
        if xf:
            n_xf += 1
            ps = prev["s"]
            tail_ss = ps.offset + prev["nf"] / fps
            cmd += ["-ss", f"{tail_ss:.3f}", "-t", f"{(xf + 3) / fps:.3f}", "-i", caches[ps.video_idx]]
            fg = (f"[0:v]split=2[o1][o2];"
                  f"[o1]trim=end_frame={xf},setpts=PTS-STARTPTS[head];"
                  f"[o2]trim=start_frame={xf},setpts=PTS-STARTPTS[rest];"
                  f"[1:v]trim=end_frame={xf},setpts=PTS-STARTPTS[tail];"
                  f"[tail][head]blend=all_expr='A*(1-(N+1)/{xf})+B*((N+1)/{xf})'[bl];"
                  f"[bl][rest]concat=n=2:v=1:a=0[v];"
                  f"[v]{','.join(chain + ['format=yuv420p'])}[out]")
            cmd += ["-filter_complex", fg, "-map", "[out]"]
        elif chain:
            cmd += ["-vf", ",".join(chain + ["format=yuv420p"])]
        cmd += ["-frames:v", str(nf), "-an", "-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf),
                "-pix_fmt", "yuv420p", "-r", str(fps), "-video_track_timescale", "90000", clip]
        jobs.append(cmd)
        clips.append(clip)

    log(f"Rendrer {len(clips)} klipp ({n_xf} med crossfade) …")
    done = [0]

    def work(cmd):
        run(cmd)
        done[0] += 1
        if done[0] % 20 == 0 or done[0] == len(jobs):
            log(f"  {done[0]}/{len(jobs)}")

    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        list(ex.map(work, jobs))

    lst = os.path.join(workdir, "list.txt")
    with open(lst, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    log("Setter sammen og legger på lyd …")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", audio_path,
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
           "-t", f"{end_time:.3f}", "-movflags", "+faststart", out_path]
    run(cmd)


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def collect_videos(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p)
                          if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".m4v")))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Lag beat-synkronisert musikkvideo av loop-videoer og en mp3.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("audio", help="mp3 (eller annen lydfil)")
    ap.add_argument("videos", nargs="+", help="videofiler og/eller mapper med videoer")
    ap.add_argument("-o", "--output", default="musikkmix.mp4")
    g = ap.add_argument_group("Klipping")
    g.add_argument("--intensity", type=float, default=1.0, help="høyere = raskere klipping (0.5–2)")
    g.add_argument("--min-cut-beats", type=int, default=1, help="korteste klipp i beats (1,2,4,8,16)")
    g.add_argument("--max-cut-beats", type=int, default=16, help="lengste klipp i beats (1,2,4,8,16)")
    g.add_argument("--min-cut-seconds", type=float, default=0.35, help="korteste klipp i sekunder")
    g.add_argument("--section-seconds", type=float, default=20.0, help="omtrentlig lengde på musikkdeler")
    g = ap.add_argument_group("Matching")
    g.add_argument("--randomness", type=float, default=0.35, help="0.05 = deterministisk, 1 = veldig tilfeldig")
    g.add_argument("--continuity", type=float, default=0.4, help="+ = like klipp etter hverandre, − = kontraster")
    g.add_argument("--pool-size", type=int, default=0, help="antall videoer per musikkgruppe (0 = auto)")
    g.add_argument("--clip", action="store_true", help="bruk CLIP (AI-modell) til visuell likhet")
    g.add_argument("--theme", default="", help="tekst-tema, f.eks. 'neon city at night' (krever CLIP)")
    g.add_argument("--seed", type=int, default=None)
    g = ap.add_argument_group("Utdata")
    g.add_argument("--resolution", default="1920x1080")
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--crf", type=int, default=18)
    g.add_argument("--preset", default="veryfast")
    g.add_argument("--drop-flash", action="store_true", help="hvit flash på drops")
    g.add_argument("--audio-offset", type=float, default=0.0, help="sek. forskyvning av klipp (+ = senere)")
    g.add_argument("--limit", type=float, default=0.0, help="lag bare de første N sekundene (test)")
    g.add_argument("--plan-only", action="store_true", help="bare skriv klippeplan (JSON), ikke render")
    g.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    g.add_argument("--keep-temp", action="store_true")
    g = ap.add_argument_group("Effekter (alle av som standard)")
    g.add_argument("--fx-preset", choices=["subtle", "punchy", "wild"],
                   help="ferdige effektsett; enkeltinnstillinger under overstyrer")
    g.add_argument("--crossfade", type=float, default=0.0, help="crossfade-varighet i sek (maks 0.5; breaks får dobbel)")
    g.add_argument("--crossfade-max-energy", type=float, default=0.65,
                   help="crossfade bare der musikkens energi er under dette (resten får hard klipp)")
    g.add_argument("--zoom", type=float, default=0.0, help="zoom-puls, andel (f.eks. 0.05 = 5 %%)")
    g.add_argument("--zoom-on", choices=["beat", "bar", "kick"], default="beat")
    g.add_argument("--shake", type=float, default=0.0, help="shake-styrke som andel av bildet (f.eks. 0.01)")
    g.add_argument("--shake-on", choices=["beat", "bar", "kick"], default="kick")
    g.add_argument("--pulse", type=float, default=0.0, help="lysstyrke-puls (f.eks. 0.12)")
    g.add_argument("--pulse-on", choices=["beat", "bar", "kick"], default="kick")
    g.add_argument("--pulse-mode", choices=["light", "dark"], default="light", help="lysere eller mørkere puls")
    a = ap.parse_args()
    presets = {
        "subtle": dict(crossfade=0.2, zoom=0.03, pulse=0.06),
        "punchy": dict(crossfade=0.12, zoom=0.05, shake=0.008, pulse=0.12),
        "wild":   dict(crossfade=0.1, zoom=0.09, shake=0.02, pulse=0.2, drop_flash=True),
    }
    if a.fx_preset:
        for key, val in presets[a.fx_preset].items():
            if not getattr(a, key):
                setattr(a, key, val)
        if a.fx_preset == "subtle" and a.zoom_on == "beat":
            a.zoom_on = "bar"
    a.crossfade = min(max(a.crossfade, 0.0), 0.5)

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"Mangler {tool}. Kjør: sudo apt install ffmpeg")
    if a.theme:
        a.clip = True
    seed = a.seed if a.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    log(f"Seed: {seed}")

    paths = collect_videos(a.videos)
    if not paths:
        sys.exit("Fant ingen videofiler.")
    workdir = tempfile.mkdtemp(prefix="musikkmix_")
    try:
        log("Analyserer lyd …")
        A = analyze_audio(a.audio, workdir, a.section_seconds)

        log(f"Analyserer {len(paths)} videoer …")
        want = 8 if a.clip else 0
        with ThreadPoolExecutor(max(1, a.jobs)) as ex:
            vids = list(ex.map(lambda p: analyze_video(p, want), paths))
        normalize_videos(vids)
        theme = np.zeros(len(vids))
        if a.clip:
            log("Kjører CLIP …")
            emb = ClipEmbedder()
            for v in vids:
                v.emb = emb.images(v.frames)
            if a.theme:
                t = emb.text(a.theme)
                s = np.array([float(v.emb @ t) for v in vids])
                theme = np.clip((s - s.mean()) / (s.std() + 1e-6), -2, 2) / 2
        for v in vids:
            log(f"  {v.name}: {v.duration:.1f}s, energi {v.energy:.2f}")
        sim = similarity_matrix(vids)

        log("Lager klippeplan …")
        slots = plan_cuts(A, a, rng)
        assign_videos(slots, A, vids, sim, a, rng, theme)
        end_time = A.duration
        if a.limit > 0:
            slots = [s for s in slots if s.t0 < a.limit]
            end_time = min(A.duration, a.limit)
        log(f"  {len(slots)} klipp, snitt {end_time / len(slots):.2f} s")

        plan_path = os.path.splitext(a.output)[0] + "_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({
                "seed": seed, "tempo": A.tempo, "duration": A.duration,
                "sections": [{"beats": [x, y], "time": [float(A.bt[x]), float(A.bt[y])], "group": A.clusters[i]}
                             for i, (x, y) in enumerate(A.sections)],
                "breaks": [[float(A.bt[x]), float(A.bt[y])] for x, y in A.breaks],
                "drops": [float(A.bt[d]) for d in A.drops], "kicks": len(A.kicks),
                "slots": [asdict(s) for s in slots],
            }, f, ensure_ascii=False, indent=1)
        log(f"Plan lagret: {plan_path}")
        if a.plan_only:
            return

        render(slots, vids, A, a.audio, a.output, a, workdir, end_time)
        log(f"Ferdig: {a.output}")
    finally:
        if a.keep_temp:
            log(f"Midlertidige filer: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
