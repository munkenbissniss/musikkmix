# musikkmix

Turn seamless video loops and an mp3 into a beat-synchronised music video. Built for big-screen visuals, runs locally on Ubuntu.

**Site:** https://YOUR-USERNAME.github.io/musikkmix/

## What it does

1. **Listens** to the mp3: tempo, beats, bars, energy per beat, song sections (grouped, so repeated parts look alike), breaks, drops and build-ups.
2. **Watches** the videos: motion over time, brightness, colour profile, optionally CLIP embeddings.
3. **Matches** loops to song sections by energy, variety and visual continuity.
4. **Cuts** on the beat: long shots in calm parts, 1–2 beat shots in intense parts, faster cuts before a drop.
5. **Renders** with ffmpeg, frame-accurate, and adds the audio.

## Install

```bash
sudo apt install ffmpeg python3-venv
git clone https://github.com/YOUR-USERNAME/musikkmix.git
cd musikkmix
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers pillow   # optional, for --clip / --theme
```

## Use

```bash
./musikkmix.py track.mp3 loops/ -o mix.mp4
./musikkmix.py track.mp3 loops/ --resolution 3840x2160 --intensity 1.3 --drop-flash
./musikkmix.py track.mp3 loops/ --clip --theme "neon city at night"
./musikkmix.py track.mp3 loops/ --limit 40       # test the first 40 s
./musikkmix.py track.mp3 loops/ --plan-only      # write the cut plan (JSON) only
```

Run `./musikkmix.py --help` for all options (`--intensity`, `--randomness`, `--continuity`, `--min-cut-beats`, `--max-cut-beats`, `--audio-offset`, `--seed`, …). The seed is printed on every run so you can reproduce a result.

## Notes

- Beat tracking is not perfect on shifting tempos or very dense rhythms. Test with `--limit` and adjust `--audio-offset` or `--seed`.
- The "AI" is signal analysis, clustering and optionally CLIP, not a language model. Everything runs locally.
- 4K with many loops needs temporary disk space.

## Publish the site

Push to GitHub, then **Settings → Pages → Deploy from a branch → `main` / `/docs`**.
Replace `YOUR-USERNAME` and `YOUR NAME` in `README.md`, `LICENSE` and `docs/index.html`.

## License

MIT
