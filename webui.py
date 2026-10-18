#!/usr/bin/env python3
"""musikkmix webgrensesnitt.   pip install flask   ->   ./webui.py   ->   http://127.0.0.1:8080
Filer og jobber lagres i ~/musikkmix-web (endres med --data). Bruk --host 0.0.0.0 for å nå det fra andre
maskiner på nettverket (ingen innlogging: bare på et nett du stoler på)."""
import argparse, json, os, queue, re, shutil, signal, subprocess, sys, threading, time, uuid
from pathlib import Path
try:
    from flask import Flask, request, jsonify, send_file, abort
    from werkzeug.utils import secure_filename
except ImportError:
    sys.exit("Mangler Flask. Kjør: pip install flask")

HERE = Path(__file__).resolve().parent
EXT = {"audio": {".mp3", ".wav", ".flac", ".m4a", ".ogg"}, "video": {".mp4", ".mov", ".mkv", ".webm", ".m4v"}}
ON = ["beat", "bar", "kick"]
def P(key, group, label, typ, default, help="", **kw): return dict(key=key, group=group, label=label, type=typ, default=default, help=help, **kw)
SCHEMA = [
 P("intensity","Klipping","Intensitet","range",1.0,"Høyere = raskere klipping.",min=0.5,max=2,step=0.05),
 P("min_cut_beats","Klipping","Korteste klipp (beats)","select",1,"",options=[1,2,4,8,16]),
 P("max_cut_beats","Klipping","Lengste klipp (beats)","select",16,"",options=[1,2,4,8,16]),
 P("min_cut_seconds","Klipping","Korteste klipp (sek)","range",0.35,"Kortere klipp forlenges.",min=0.1,max=1.5,step=0.05),
 P("section_seconds","Klipping","Delens lengde (sek)","range",20,"Omtrentlig lengde på musikkdeler. Mindre = flere skifter i look.",min=8,max=60,step=1),
 P("randomness","Matching","Tilfeldighet","range",0.35,"Lav = forutsigbart, høy = fritt.",min=0.05,max=1,step=0.05),
 P("continuity","Matching","Kontinuitet","range",0.4,"Positiv = like klipp etter hverandre, negativ = kontraster.",min=-1,max=1,step=0.05),
 P("pool_size","Matching","Videoer per musikkgruppe","range",0,"0 = automatisk.",min=0,max=12,step=1),
 P("clip","Matching","Bruk CLIP (AI-bildeforståelse)","bool",False,"Krever torch, transformers og pillow."),
 P("theme","Matching","Tema (engelsk tekst)","text","","F.eks. neon city at night. Slår på CLIP."),
 P("seed","Matching","Seed (tom = tilfeldig)","number","","Samme seed gir samme resultat."),
 P("crossfade","Effekter","Crossfade (sek)","range",0,"Bare i rolige partier. 0 = av.",min=0,max=0.5,step=0.05),
 P("crossfade_max_energy","Effekter","Crossfade under energi","range",0.65,"",min=0,max=1,step=0.05),
 P("zoom","Effekter","Zoom-puls","range",0,"0 = av. 0.05 = 5 %.",min=0,max=0.12,step=0.005),
 P("zoom_on","Effekter","Zoom på","select","beat","",options=ON),
 P("shake","Effekter","Shake","range",0,"0 = av.",min=0,max=0.03,step=0.001),
 P("shake_on","Effekter","Shake på","select","kick","",options=ON),
 P("pulse","Effekter","Lysstyrke-puls","range",0,"0 = av.",min=0,max=0.3,step=0.01),
 P("pulse_on","Effekter","Puls på","select","kick","",options=ON),
 P("pulse_mode","Effekter","Pulstype","select","light","",options=["light","dark"]),
 P("drop_flash","Effekter","Hvit flash på drops","bool",False),
 P("resolution","Utdata","Oppløsning","select","1920x1080","",options=["1280x720","1920x1080","2560x1440","3840x2160","1080x1920"]),
 P("fps","Utdata","Bilder per sekund","select",30,"",options=[24,25,30,50,60]),
 P("crf","Utdata","Kvalitet (CRF, lavere = bedre)","range",18,"",min=12,max=28,step=1),
 P("preset","Utdata","Hastighet (x264)","select","veryfast","Tregere = mindre fil.",options=["ultrafast","veryfast","fast","medium","slow"]),
 P("audio_offset","Utdata","Forskyv klipp (sek)","range",0,"+ = senere, − = tidligere.",min=-0.2,max=0.2,step=0.005),
]
PRESETS = {"Ingen": {}, "Subtle": dict(crossfade=0.2, zoom=0.03, zoom_on="bar", pulse=0.06),
           "Punchy": dict(crossfade=0.12, zoom=0.05, shake=0.008, pulse=0.12),
           "Wild": dict(crossfade=0.1, zoom=0.09, shake=0.02, pulse=0.2, drop_flash=True)}
BYKEY = {p["key"]: p for p in SCHEMA}

ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--data", default=str(Path.home() / "musikkmix-web")); ARGS = ap.parse_args()
DATA = Path(ARGS.data); LIB = DATA / "library"; JD = DATA / "jobs"
for k in EXT: (LIB / k).mkdir(parents=True, exist_ok=True)
JD.mkdir(exist_ok=True)
app = Flask(__name__); JOBS, PROCS, Q, LOCK = {}, {}, queue.Queue(), threading.Lock()

def save(j): (JD / j["id"] / "job.json").write_text(json.dumps(j, ensure_ascii=False))
for f in sorted(JD.glob("*/job.json")):
    j = json.loads(f.read_text())
    if j["status"] in ("queued", "running"): j["status"], j["error"] = "failed", "Avbrutt av omstart"
    JOBS[j["id"]] = j

def build_args(params, test, plan_only):
    args = []
    for key, v in params.items():
        p = BYKEY.get(key)
        if not p or v in (None, ""): continue
        flag = "--" + key.replace("_", "-")
        if p["type"] == "bool":
            if v is True: args.append(flag)
        elif p["type"] == "text": args += [flag, str(v).strip()[:200]]
        elif p["type"] == "select":
            if str(v) not in [str(o) for o in p["options"]]: raise ValueError(key)
            args += [flag, str(v)]
        else:
            x = float(v)
            if p["type"] == "range" and not p["min"] <= x <= p["max"]: raise ValueError(key)
            args += [flag, str(int(x)) if p["step"] >= 1 or key == "seed" else str(x)] if p["type"] == "range" else [flag, str(int(x))]
    if test: args += ["--limit", str(int(test))]
    if plan_only: args.append("--plan-only")
    return args

STAGES = [("Analyserer lyd", 3), ("videoer …", 12), ("Lager klippeplan", 26), ("Normaliserer", 30), ("Rendrer", 45), ("Setter sammen", 92)]
def progress(text):
    pct, stage = 0, "Starter"
    for line in text.splitlines():
        for key, v in STAGES:
            if key in line: pct, stage = v, line.strip().rstrip(" …")
        m = re.match(r"\s+(\d+)/(\d+)$", line)
        if m: pct = 45 + 45 * int(m[1]) / int(m[2]); stage = f"Rendrer klipp {m[1]}/{m[2]}"
        if line.startswith(("Ferdig", "Plan lagret")): pct, stage = 100, line.strip()
    return round(pct), stage

def view(j, tail=False):
    d = JD / j["id"]; log = (d / "log.txt").read_text(errors="replace") if (d / "log.txt").exists() else ""
    out = dict(j, has_video=(d / "mix.mp4").exists(), has_plan=(d / "mix_plan.json").exists())
    out["progress"], out["stage"] = progress(log)
    m = re.search(r"Seed: (\d+)", log); out["seed"] = int(m[1]) if m else None
    out["log"] = log[-6000:]
    if j["status"] == "done": out["progress"] = 100
    return out

def worker():
    while True:
        jid = Q.get(); j = JOBS.get(jid)
        if not j or j["status"] != "queued": continue
        j["status"] = "running"; save(j); d = JD / jid
        cmd = [sys.executable, str(HERE / "musikkmix.py"), str(LIB / "audio" / j["audio"]),
               *[str(LIB / "video" / v) for v in j["videos"]], "-o", str(d / "mix.mp4"), *j["args"]]
        with open(d / "log.txt", "w") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            PROCS[jid] = p; rc = p.wait()
        if j["status"] != "cancelled": j["status"] = "done" if rc == 0 else "failed"
        save(j)
threading.Thread(target=worker, daemon=True).start()

@app.get("/")
def index(): return send_file(HERE / "web" / "index.html")
@app.get("/api/schema")
def schema(): return jsonify(schema=SCHEMA, presets=PRESETS, ffmpeg=bool(shutil.which("ffmpeg")))
@app.get("/api/library")
def library():
    return jsonify({k: [dict(name=f.name, size=f.stat().st_size) for f in sorted((LIB / k).iterdir())] for k in EXT})
@app.post("/api/library")
def upload():
    f = request.files.get("file"); name = secure_filename(f.filename or "") if f else ""
    kind = next((k for k, e in EXT.items() if Path(name).suffix.lower() in e), None)
    if not kind: return jsonify(error="Filtypen støttes ikke"), 400
    f.save(LIB / kind / name); return jsonify(ok=True, kind=kind, name=name)
def lib_path(kind, name):
    if kind not in EXT or name != secure_filename(name) or not (LIB / kind / name).is_file(): abort(404)
    return LIB / kind / name
@app.get("/api/library/<kind>/<name>")
def lib_get(kind, name): return send_file(lib_path(kind, name), conditional=True)
@app.delete("/api/library/<kind>/<name>")
def lib_del(kind, name): lib_path(kind, name).unlink(); return jsonify(ok=True)

@app.post("/api/jobs")
def new_job():
    b = request.get_json(force=True)
    audio, videos = b.get("audio", ""), b.get("videos", [])
    if not audio or not videos: return jsonify(error="Velg én lydfil og minst én video"), 400
    try:
        lib_path("audio", audio); [lib_path("video", v) for v in videos]
        args = build_args(b.get("params", {}), b.get("test"), b.get("plan_only"))
    except ValueError as e: return jsonify(error=f"Ugyldig verdi: {e}"), 400
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]; (JD / jid).mkdir()
    j = dict(id=jid, created=time.time(), status="queued", audio=audio, videos=videos, args=args,
             params=b.get("params", {}), test=b.get("test"), plan_only=bool(b.get("plan_only")))
    JOBS[jid] = j; save(j); Q.put(jid); return jsonify(id=jid)
@app.get("/api/jobs")
def jobs(): return jsonify([view(j) for j in sorted(JOBS.values(), key=lambda j: -j["created"])])
@app.get("/api/jobs/<jid>/video")
def job_video(jid):
    f = JD / jid / "mix.mp4"
    if jid not in JOBS or not f.exists(): abort(404)
    return send_file(f, conditional=True, as_attachment=bool(request.args.get("download")), download_name=f"musikkmix-{jid}.mp4")
@app.get("/api/jobs/<jid>/plan")
def job_plan(jid):
    f = JD / jid / "mix_plan.json"
    if jid not in JOBS or not f.exists(): abort(404)
    return send_file(f, as_attachment=True, download_name=f"musikkmix-{jid}-plan.json")
@app.post("/api/jobs/<jid>/cancel")
def cancel(jid):
    j = JOBS.get(jid) or abort(404)
    if j["status"] in ("queued", "running"):
        j["status"] = "cancelled"; save(j)
        if jid in PROCS and PROCS[jid].poll() is None: os.killpg(PROCS[jid].pid, signal.SIGTERM)
    return jsonify(ok=True)
@app.delete("/api/jobs/<jid>")
def del_job(jid):
    j = JOBS.get(jid) or abort(404)
    if j["status"] in ("queued", "running"): return jsonify(error="Avbryt jobben først"), 400
    JOBS.pop(jid); shutil.rmtree(JD / jid, ignore_errors=True); return jsonify(ok=True)

if __name__ == "__main__":
    print(f"musikkmix web: http://{ARGS.host}:{ARGS.port}   (data: {DATA})")
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)
