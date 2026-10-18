# musikkmix

Turn seamless video loops and an mp3 into a beat-synchronised music video. Built for big-screen visuals, runs locally on Ubuntu.

**Site:** https://YOUR-USERNAME.github.io/musikkmix/

## Contents

- [What it does](#what-it-does)
- [Install](#install)
- [Quick start](#quick-start)
- [Web interface](#web-interface)
- [Parameter reference](#parameter-reference)
  - [Input and output](#input-and-output)
  - [Cutting](#cutting)
  - [Matching](#matching)
  - [Video output](#video-output)
  - [Effects](#effects)
  - [Utility](#utility)
- [Effect presets](#effect-presets)
- [Recipes](#recipes)
- [How the analysis works](#how-the-analysis-works)
- [The plan file](#the-plan-file)
- [Troubleshooting](#troubleshooting)
- [Publish the site](#publish-the-site)
- [License](#license)

## What it does

1. **Listens** to the mp3: tempo, beats, bars, energy per beat, song sections (grouped, so repeated parts look alike), breaks, drops, build-ups and kick drums.
2. **Watches** the videos: motion over time, brightness, colour profile, optionally CLIP embeddings.
3. **Matches** loops to song sections by energy, variety and visual continuity.
4. **Cuts** on the beat: long shots in calm parts, 1-2 beat shots in intense parts, faster cuts before a drop.
5. **Adds effects** (optional): crossfades in calm parts, zoom, shake and brightness pulses locked to beats, bars or kicks.
6. **Renders** with ffmpeg, frame-accurate, and adds the audio.

The input videos are expected to be **seamless loops**. musikkmix starts shots at different points inside each loop and loops them as needed, so a loop that does not repeat cleanly will show a jump.

## Install

```bash
sudo apt install ffmpeg python3-venv
git clone https://github.com/YOUR-USERNAME/musikkmix.git
cd musikkmix
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers pillow   # optional, only for --clip / --theme
```

Requirements: Ubuntu (or another Linux with ffmpeg 4.4 or newer), Python 3.9+. The `--clip` model (`openai/clip-vit-base-patch32`) is downloaded on first use, which needs an internet connection once.

## Quick start

```bash
./musikkmix.py track.mp3 loops/ -o mix.mp4
./musikkmix.py track.mp3 loops/ --fx-preset punchy
./musikkmix.py track.mp3 loops/ --limit 40          # test the first 40 s
./musikkmix.py track.mp3 loops/ --plan-only         # only write the cut plan
```

Always test a new track with `--limit 40` first. It is much faster and shows whether the beat tracking and effect timing feel right.

## Web interface

A browser interface for people who prefer clicking to typing commands.

```bash
pip install flask
./webui.py                       # then open http://127.0.0.1:8080
./webui.py --host 0.0.0.0        # reachable from other machines on your network (no login!)
./webui.py --port 9000 --data /big/disk/musikkmix-web
```

1. **Music and loops:** drag files into the page, or use the file buttons. Uploads go to a library (`~/musikkmix-web/library`), so you only upload each loop once and can reuse it in later mixes. Loops show as thumbnails that play on hover. Click to select, and use the cross to delete.
2. **Settings:** every parameter from the reference below is a slider, dropdown or checkbox with a short explanation. The effect buttons (Subtle, Punchy, Wild) fill in the effect controls, which you can then adjust.
3. **Make the video:** *Test* renders only the first N seconds of the track (fast, good for checking timing). *Bare analyse* only writes the plan and skips rendering.
4. **Jobs:** each run shows progress, a log, a video player and a download link. *Gjenbruk innstillinger* loads a finished job's settings back into the form, and *Bruk seed* locks its random seed so you can tweak one setting and compare. Jobs run one at a time, and you can cancel them.

The server has no password. Keep the default `127.0.0.1` unless you trust everyone on your network.

## Parameter reference

```
./musikkmix.py AUDIO VIDEO [VIDEO ...] [options]
```

Defaults are shown in the tables. `./musikkmix.py --help` lists the same options (the built-in help text is in Norwegian).

### Input and output

| Parameter | Default | Description |
|---|---|---|
| `AUDIO` (positional) | required | The music. Any audio file ffmpeg can decode (mp3, wav, flac, m4a, ...). |
| `VIDEO ...` (positional) | required | One or more video files and/or folders. A folder is scanned (not recursively) for `.mp4`, `.mov`, `.mkv`, `.webm`, `.m4v`. Files and folders can be mixed. More loops with distinct looks give a more varied result. |
| `-o`, `--output` | `musikkmix.mp4` | Output file. A plan file named `<output>_plan.json` is written next to it. |

### Cutting

These control how often the picture changes. All cuts land on the beat grid.

| Parameter | Default | Description |
|---|---|---|
| `--intensity` | `1.0` | Multiplies the music's energy before choosing shot lengths. Above 1 gives faster cutting, below 1 slower. Useful range 0.5-2. Roughly: energy above 0.85 gives 1-beat shots, above 0.65 gives 2, above 0.4 gives 4, above 0.2 gives 8, otherwise 16. About a quarter of shots are moved one step up or down for variety. |
| `--min-cut-beats` | `1` | Shortest allowed shot in beats. Valid values are 1, 2, 4, 8, 16; other numbers snap to the nearest. Use `2` to calm down very busy tracks. |
| `--max-cut-beats` | `16` | Longest allowed shot in beats (1, 2, 4, 8, 16). Use `8` to avoid long static shots in breaks. |
| `--min-cut-seconds` | `0.35` | Shots shorter than this are lengthened. Matters at high BPM, where a single beat can be very short. Lower it for frantic cutting, raise it for calmer results. |
| `--section-seconds` | `20.0` | Approximate length of the song sections the analysis looks for. Smaller values find more, shorter sections (more changes of look), larger values find fewer. Sections are snapped to bar lines and are at least 8 beats long. |

How shot lengths are chosen:

- A shot is always aligned to the bar grid. For example, a 4-beat shot can only start where a 4-beat boundary falls, so cutting stays rhythmic instead of random.
- **Breaks** (strong energy dips) get shots of at least 8 beats.
- **Build-ups** (the 8 beats before a drop) get progressively shorter shots.
- A **drop** gets a hard cut exactly on the downbeat, using the most energetic loop available.
- Shots never run across a section boundary.

### Matching

These control which loop is used where.

| Parameter | Default | Description |
|---|---|---|
| `--randomness` | `0.35` | How adventurous the loop choice is. `0.05` always picks the best match (predictable), `1` picks almost freely. |
| `--continuity` | `0.4` | How much the next shot should look like the previous one. Positive values favour smooth visual flow, `0` ignores it, negative values favour contrast between consecutive shots. |
| `--pool-size` | `0` (auto) | How many loops each group of similar song sections may draw from. Smaller means each part of the song has a stricter, more recognisable look. `0` picks a value from the number of loops and song groups. |
| `--clip` | off | Use the CLIP AI model to judge what the loops actually show (instead of only colour and motion) when computing visual similarity. Needs `torch`, `transformers`, `pillow`. Slower to start, noticeably better continuity with varied footage. |
| `--theme TEXT` | none | A text prompt such as `"neon city at night"`. Loops that match the text get a bonus. Implies `--clip`. Works best with short, concrete descriptions in English. |
| `--seed N` | random | Seed for all random choices. The seed is printed on every run. Reuse it to reproduce a result exactly, change it to get a different variation of the same settings. |

Loop energy is measured as motion in the picture. Calm sections of the song prefer calm loops and intense sections prefer busy ones. Within a loop, the start point is also chosen so the motion at that spot fits the music, and repeated use of the same spot is avoided.

### Video output

| Parameter | Default | Description |
|---|---|---|
| `--resolution` | `1920x1080` | Output size as `WIDTHxHEIGHT`, for example `3840x2160` or `1080x1920`. Loops are scaled to fill the frame and centre-cropped (no black bars). Odd numbers are rounded down to even. |
| `--fps` | `30` | Output frame rate. Cuts are placed on exact frame numbers. |
| `--crf` | `18` | H.264 quality. Lower is better and larger: 14-18 is high quality, 23 is the x264 default. |
| `--preset` | `veryfast` | x264 speed preset: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`. Slower presets give smaller files at the same quality. |
| `--drop-flash` | off | A quick white flash (0.25 s fade-in from white) on the first shot of every drop. |
| `--audio-offset` | `0.0` | Shifts all cuts **and** effects by this many seconds. Positive means later, negative earlier. Use it if everything feels consistently a little early or late (for example `-0.04` or `0.03`). |
| `--limit SEC` | `0` (all) | Only render the first SEC seconds. Great for testing. |

The audio is encoded as AAC at 320 kbit/s. The final video is assembled without re-encoding the clips.

### Effects

All effects are **off by default**. Each effect pulses at a trigger and then fades out. Triggers:

- `beat`: every beat.
- `bar`: the first beat of every bar (every 4th beat).
- `kick`: detected kick drums, i.e. sudden rises in energy below 150 Hz. Stronger kicks give stronger effects.

| Parameter | Default | Description |
|---|---|---|
| `--fx-preset` | none | `subtle`, `punchy` or `wild`. See [Effect presets](#effect-presets). Any effect you set yourself wins over the preset. |
| `--crossfade SEC` | `0.0` (off) | Dissolve into the next shot over SEC seconds. Capped at 0.5; doubled for shots inside breaks. The dissolve starts on the beat. Drops are never crossfaded. Typical values: 0.1-0.3. |
| `--crossfade-max-energy` | `0.65` | Crossfades are only used where the music's energy is at or below this value (0-1). Intense parts keep hard cuts. Set to `1.0` to crossfade everywhere (except drops), or `0.3` to only crossfade very calm parts. |
| `--zoom` | `0.0` (off) | Zoom pulse: how far the picture zooms in at the peak, as a fraction. `0.05` means 5 %. Typical values: 0.03-0.09. Decays in about 0.15 s. |
| `--zoom-on` | `beat` | Trigger for the zoom: `beat`, `bar` or `kick`. |
| `--shake` | `0.0` (off) | Camera shake amplitude as a fraction of the frame size. `0.01` moves the picture up to 1 % of its width/height. Typical values: 0.005-0.02. Decays very quickly (about 0.07 s). The picture is kept slightly zoomed in (2 times the value) so edges never show. |
| `--shake-on` | `kick` | Trigger for the shake: `beat`, `bar` or `kick`. |
| `--pulse` | `0.0` (off) | Brightness pulse strength (0-1). Typical values: 0.06-0.2. Decays in about 0.1 s. |
| `--pulse-on` | `kick` | Trigger for the pulse: `beat`, `bar` or `kick`. |
| `--pulse-mode` | `light` | `light` flashes brighter on the trigger, `dark` dips darker. |

Details:

- On **drops**, zoom and pulse are 1.6 times stronger.
- Zoom, shake and pulse can use different triggers at the same time, for example zoom on every bar, pulse on kicks.
- Effects add rendering time. Zoom and shake are the most expensive in 4K.
- If effects feel slightly off the beat, adjust `--audio-offset`.

### Utility

| Parameter | Default | Description |
|---|---|---|
| `--plan-only` | off | Analyse and plan, write the plan JSON, but do not render. Takes seconds and shows what the program found. |
| `--jobs N` | half the CPU cores | Number of videos analysed and clips rendered in parallel. Each ffmpeg process also uses several threads itself, so more is not always faster. Lower it if the machine runs out of memory. |
| `--keep-temp` | off | Keep the temporary working folder (normalised loops and individual clips) and print its path. Handy for debugging. It is deleted by default. |

## Effect presets

Presets fill in effects you have not set yourself.

| Preset | Crossfade | Zoom | Shake | Pulse | Other |
|---|---|---|---|---|---|
| `subtle` | 0.2 s | 0.03 on `bar` | off | 0.06 on `kick` | |
| `punchy` | 0.12 s | 0.05 on `beat` | 0.008 on `kick` | 0.12 on `kick` | |
| `wild` | 0.1 s | 0.09 on `beat` | 0.02 on `kick` | 0.2 on `kick` | `--drop-flash` |

Examples:

```bash
./musikkmix.py track.mp3 loops/ --fx-preset punchy --pulse-mode dark     # preset, but dark pulses
./musikkmix.py track.mp3 loops/ --fx-preset wild --zoom 0.05             # preset, weaker zoom
```

A value you pass always wins over the preset. Because an effect value of `0` means "off / not set", you cannot switch off a preset effect by passing `0`; use individual options instead of a preset in that case.

## Recipes

Calm, flowing visuals for a chill-out set:

```bash
./musikkmix.py track.mp3 loops/ --intensity 0.7 --min-cut-beats 4 \
  --crossfade 0.3 --crossfade-max-energy 1.0 --continuity 0.8 --randomness 0.2
```

Hard-hitting techno or drum and bass:

```bash
./musikkmix.py track.mp3 loops/ --intensity 1.4 --fx-preset punchy --drop-flash
```

Big screen at 4K, final quality:

```bash
./musikkmix.py track.mp3 loops/ --resolution 3840x2160 --crf 16 --preset medium --jobs 4
```

Themed mix with AI matching:

```bash
./musikkmix.py track.mp3 loops/ --clip --theme "abstract liquid colours" --continuity 0.6
```

Same settings, different variation (keep your favourite by noting the seed):

```bash
./musikkmix.py track.mp3 loops/ --seed 1
./musikkmix.py track.mp3 loops/ --seed 2
```

Fast iteration on timing:

```bash
./musikkmix.py track.mp3 loops/ --limit 30 --audio-offset -0.03 --preset ultrafast
```

## How the analysis works

**Audio.** The track is decoded to mono and analysed with librosa.

- *Beats and tempo:* beat tracking on the onset-strength curve. The bar's "one" (the downbeat) is guessed as the beat position with the strongest average onset.
- *Energy:* per beat, a mix of loudness (55 %), onset strength (30 %) and spectral brightness (15 %), smoothed and scaled 0-1.
- *Sections:* agglomerative segmentation of MFCC and chroma features, snapped to bar lines. Sections are then clustered with k-means, so similar parts (for example every chorus) end up in the same group and get the same look.
- *Breaks:* runs of at least 4 beats where energy falls below 45 % of the surrounding maximum.
- *Drops:* sudden energy rises (the next 4 beats compared with the previous 4), snapped to a downbeat and at least 16 beats apart.
- *Kicks:* peaks in the sudden rise of energy below 150 Hz, at least 0.12 s apart.

**Video.** Each loop is sampled at about 4 frames per second. Motion, brightness, saturation and a hue histogram are measured. With `--clip`, up to 8 frames per loop are embedded with CLIP and averaged. Motion is normalised across all your loops, so the calmest loop counts as calm and the busiest as intense, whatever their absolute values.

**Rendering.** Each loop is first converted to the target resolution and frame rate and extended by looping, so any start point works. Every shot is rendered with its effects, shots are joined without re-encoding, and the audio is added.

The "AI" here is signal analysis, clustering and optionally CLIP. It is not a language model and nothing is sent to any server (except the one-time model download for `--clip`).

## The plan file

Each run writes `<output>_plan.json` next to the output video:

```json
{
  "seed": 123456, "tempo": 128.0, "duration": 215.3,
  "sections": [{"beats": [0, 32], "time": [0.0, 15.0], "group": 0}],
  "breaks": [[91.2, 106.1]],
  "drops": [106.1],
  "kicks": 412,
  "slots": [
    {"i0": 0, "i1": 4, "t0": 0.0, "t1": 1.88, "energy": 0.31, "target": 0.31,
     "section": 0, "cluster": 0, "kind": "normal",
     "video": "waves.mp4", "video_idx": 2, "offset": 1.5}
  ]
}
```

`kind` is `normal`, `break`, `buildup` or `drop`. `offset` is the start point inside the loop, in seconds. Use `--plan-only` to inspect the plan quickly before committing to a full render.

## Troubleshooting

**"Fant for få beats i lydfilen"** (too few beats found). The track is not rhythmic enough for the beat tracker. Try a track with a clearer pulse.

**Cuts feel slightly early or late.** Use `--audio-offset` in small steps (0.02-0.05 s). MP3 decoding delay and display latency can shift things a little.

**Cuts land on the wrong beat in places.** Beat tracking can drift on tracks with tempo changes, live drums or very dense rhythms. Try another `--seed`, or edit the result around the problem section.

**Too frantic.** Lower `--intensity`, set `--min-cut-beats 2`, or raise `--min-cut-seconds`.

**Too static.** Raise `--intensity`, set `--max-cut-beats 8`, or lower `--section-seconds` for more changes of look.

**The same loop appears too often.** Add more loops, raise `--randomness`, or raise `--pool-size`.

**Everything looks the same.** Lower `--continuity` (even negative), raise `--pool-size`, or use loops with more varied motion and colour.

**Rendering is slow.** Use `--preset ultrafast`, a lower `--resolution`, fewer effects (zoom and shake cost most), or test with `--limit`.

**Out of memory or disk.** Lower `--jobs`. Temporary files are stored in `/tmp` and can become large with many 4K loops; the folder is removed when the run finishes.

**`--clip` fails to import.** Install the optional packages: `pip install torch transformers pillow`.

## Publish the site

Push to GitHub, then **Settings -> Pages -> Deploy from a branch -> `main` / `/docs`**.
Replace `YOUR-USERNAME` and `YOUR NAME` in `README.md`, `LICENSE` and `docs/index.html`.

## License

MIT
