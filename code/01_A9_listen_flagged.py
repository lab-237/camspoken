# 01_A9_listen_flagged
# AIM: Manual audio audit: listen to every clip next to its own sentence, with a moving waveform; Pidgin and Camfranglais controlled separately
# saved: 2026-10-24 18:57

# A9: manual audio audit (listening check). Needs 00a to have run (and content_check.csv from 01_A8 for the "flagged" buttons).
# One panel per language. Each panel has its own ID box, Previous/Next, and a selection box (e.g. 1-18, 25, 40-44).
# Only the clip on screen is loaded, so nothing waits for all 228 files.
import json, base64
import numpy as np, pandas as pd, soundfile as sf, librosa
from IPython.display import HTML, JSON, display
from google.colab import output

_A9_DF = {"pidgin": pidgin_df, "camfranglais": camf_df}

# items flagged by the content check (01_A8), per language: a one-click selection in each panel
_A9_FLAGS, _A9_FLAGGED = {}, {"pidgin": [], "camfranglais": []}
_chk_path = RESULTS_DIR / "content_check.csv"
if _chk_path.exists():
    _chk = pd.read_csv(_chk_path).fillna("")
    for _, r in _chk[_chk["flag"] != ""].iterrows():
        _A9_FLAGS[r["id"]] = str(r["flag"])
        _A9_FLAGGED[r["variety"]].append(int(r["id"][-5:]))
for v in _A9_FLAGGED:
    _A9_FLAGGED[v] = sorted(_A9_FLAGGED[v])


def _a9_item(variety, n):
    """Called from the page: return the text, the audio (base64 WAV) and a pitch track for ONE clip."""
    n = int(n)
    item_id = f"{ID_PREFIX[variety]}_{n:05d}"
    df = _A9_DF[variety].set_index("id")
    if item_id not in df.index:
        return JSON({"error": f"{item_id} is not in the CSV"})
    row = df.loc[item_id]
    wav = AUDIO_OUT_DIR / variety / f"{item_id}.wav"
    if not wav.exists():
        return JSON({"error": f"audio file not found: {wav}"})
    y, sr = sf.read(wav, dtype="float32")
    # pitch track (indicative only): 100 values per second, 0 = no voice
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=160)[0]
    m = min(len(f0), len(rms))
    f0 = np.where(rms[:m] > 0.15 * rms[:m].max(), f0[:m], 0.0)
    return JSON({
        "id": item_id,
        "csv_audio_path": str(row["premise_audio"]),
        "audio_source": wav.name,
        "premise": str(row["premise_text"]),
        "gloss": str(row["premise_text_en"]),
        "label": str(row["label"]),
        "flag": _A9_FLAGS.get(item_id, ""),
        "duration": float(len(y) / sr),
        "f0": [round(float(x), 1) for x in f0],
        "wav_b64": base64.b64encode(wav.read_bytes()).decode(),
    })


output.register_callback("a9.item", _a9_item)

_A9_HTML = r"""
<style>
  .a9 { --ink:#111827; --mute:#374151; --line:#9aa5b5; --accent:#1d4ed8; --wave:#b9c3d3; --played:#2b5fb4; --pitch:#d9731a; --btn:#e8ecf2; --bg:#f6f7f9;
        font:16px/1.5 system-ui, sans-serif; color:var(--ink); max-width:780px; }
  @media (prefers-color-scheme: dark) { .a9 { --ink:#f3f5f8; --mute:#c3cad6; --line:#4a5568; --accent:#8db4ff; --wave:#3b4658; --played:#79a6ee; --pitch:#f0a35a; --btn:#2a3240; --bg:#1b212b; } }
  .a9 * { box-sizing:border-box; }
  .a9 h1 { font-size:20px; margin:4px 0 2px; } .a9 .sub { font-weight:600; margin-bottom:12px; }
  .a9 .panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:18px; }
  .a9 h2 { font-size:15px; margin:0 0 10px; } .a9 h2 span { font-weight:400; color:var(--mute); }
  .a9 .row { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  .a9 .row label { font-size:13px; color:var(--mute); min-width:56px; }
  .a9 input[type=text] { padding:7px 9px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); font:inherit; }
  .a9 .idbox { width:70px; } .a9 .selbox { flex:1; min-width:150px; }
  .a9 button { padding:7px 14px; border:1px solid var(--line); border-radius:6px; background:var(--btn); color:var(--ink); font:inherit; cursor:pointer; }
  .a9 button:hover { border-color:var(--accent); } .a9 button.grow { flex:1; }
  .a9 .status { font-size:14px; color:var(--mute); margin:2px 0 10px; }
  .a9 .details { font:15px/1.6 ui-monospace, Menlo, monospace; border-top:1px dashed var(--line); border-bottom:1px dashed var(--line); padding:10px 0; margin-bottom:12px; }
  .a9 .details div { display:flex; gap:8px; } .a9 .details b { color:var(--mute); font-weight:500; min-width:150px; flex:none; }
  .a9 .details .warn { color:#c0392b; }
  .a9 .audio { display:flex; gap:10px; align-items:flex-start; }
  .a9 .play { width:40px; height:40px; border-radius:50%; border:0; background:var(--accent); color:#fff; font-size:15px; padding:0; flex:none; margin-top:20px; }
  .a9 .wavebox { flex:1; min-width:0; }
  .a9 canvas { width:100%; height:110px; display:block; border:1px solid var(--line); border-radius:6px; background:var(--bg); cursor:pointer; }
  .a9 .times { display:flex; justify-content:space-between; font:14px ui-monospace, monospace; color:var(--mute); margin-top:4px; flex-wrap:wrap; gap:6px; }
  .a9 .times b { color:var(--ink); }
  .a9 .opts { font-size:14px; color:var(--mute); margin-top:6px; display:flex; gap:14px; flex-wrap:wrap; }
  .a9 .opts label { display:flex; gap:5px; align-items:center; cursor:pointer; }
</style>
<div class="a9">
  <h1>CamSpokenNLI (v1.0) - Audio Audit</h1>
  <div class="sub">Each language is controlled independently.</div>
  <div class="panel" id="a9_pidgin"></div>
  <div class="panel" id="a9_camfranglais"></div>
</div>
<script>
(function () {
const FLAGGED = __FLAGGED__;
const LANGS = { pidgin: "CAMEROON PIDGIN", camfranglais: "CAMFRANGLAIS" };
let AC = null, active = null;
const ctx = () => (AC = AC || new (window.AudioContext || window.webkitAudioContext)());
const ALL = Array.from({length:114}, (_, i) => i + 1);

function build(variety) {
  const root = document.getElementById("a9_" + variety), flagged = FLAGGED[variety] || [];
  const st = { n: 1, sel: ALL.slice(), item: null, buf: null, peaks: null, playing: false, offset: 0, t0: 0, src: null, token: 0 };
  root.innerHTML = `
    <h2><span class="ttl"></span></h2>
    <div class="row"><label>ID</label><input type="text" class="idbox" value="1">
      <button class="go">Go</button><button class="grow prev">&#9664; Previous</button><button class="grow next">Next &#9654;</button></div>
    <div class="row"><label>Select</label><input type="text" class="selbox" placeholder="e.g. 1-18, 25, 40-44">
      <button class="apply">Use selection</button><button class="fl">Flagged only (${flagged.length})</button><button class="all">All 114</button></div>
    <div class="status"></div>
    <div class="details"></div>
    <div class="audio"><button class="play">&#9654;</button>
      <div class="wavebox"><canvas></canvas>
        <div class="times"><span>Position <b class="pos">0.000</b> s</span><span>Total duration <b class="tot">0.000</b> s</span></div>
        <div class="opts"><label><input type="checkbox" class="pt"> show pitch line (indicative)</label><span>click the waveform to jump</span></div>
      </div></div>`;
  const $ = q => root.querySelector(q), cv = $("canvas"), g = cv.getContext("2d");
  const css = v => getComputedStyle(root.firstElementChild || root).getPropertyValue(v).trim() || getComputedStyle(document.querySelector(".a9")).getPropertyValue(v).trim();
  const css2 = v => getComputedStyle(document.querySelector(".a9")).getPropertyValue(v).trim();
  const pos = () => st.playing ? Math.min(st.buf.duration, ctx().currentTime - st.t0) : st.offset;

  function line(label, text, cls) { const d = document.createElement("div"), b = document.createElement("b"), s = document.createElement("span");
    b.textContent = label; s.textContent = text; if (cls) s.className = cls; d.append(b, s); return d; }

  async function load(n) {
    const token = ++st.token; stop(true); st.n = n; $(".idbox").value = n;
    $(".ttl").innerHTML = `${LANGS[variety]} <span>— Item ${n}/114</span>`;
    const idx = st.sel.indexOf(n);
    $(".status").textContent = (idx >= 0 ? `Selection: item ${idx + 1} of ${st.sel.length}` + (st.sel.length === 114 ? " (all items)" : "")
                                         : `Item ${n} is outside the current selection (${st.sel.length} items)`) + "  · loading…";
    let it;
    try {
      const r = await google.colab.kernel.invokeFunction("a9.item", [variety, n], {});
      it = r.data["application/json"];
    } catch (e) { $(".status").textContent = "Could not load item " + n + ": " + e; return; }
    if (token !== st.token) return;                       // a newer click replaced this one
    if (it.error) { $(".status").textContent = it.error; return; }
    let buf;
    try {
      const bytes = Uint8Array.from(atob(it.wav_b64), c => c.charCodeAt(0));
      buf = await ctx().decodeAudioData(bytes.buffer);
    } catch (e) { $(".status").textContent = "Could not decode the audio of item " + n + ": " + e; return; }
    if (token !== st.token) return;
    st.item = it; st.buf = buf; st.offset = 0;
    $(".status").textContent = $(".status").textContent.replace("  · loading…", "");
    const d = $(".details"); d.innerHTML = "";
    d.append(line("ID:", it.id), line("CSV audio path:", it.csv_audio_path), line("Audio source:", it.audio_source),
             line("PREMISE:", it.premise), line("ENGLISH GLOSS:", it.gloss), line("LABEL:", it.label));
    if (it.flag) d.append(line("CONTENT CHECK:", "flagged: " + it.flag + "  (listen carefully)", "warn"));
    $(".tot").textContent = buf.duration.toFixed(3); $(".pos").textContent = "0.000";
    draw();
  }

  function draw() {
    if (!st.buf) return;
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 600, H = cv.clientHeight || 110;
    cv.width = W * dpr; cv.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const y = st.buf.getChannelData(0), dur = st.buf.duration, px = pos() / dur * W, mid = H / 2, step = y.length / W;
    for (let x = 0; x < W; x++) {
      let mn = 0, mx = 0; const a = Math.floor(x * step), b = Math.min(y.length, Math.floor((x + 1) * step));
      for (let i = a; i < b; i++) { const v = y[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      g.fillStyle = x < px ? css2("--played") : css2("--wave");
      g.fillRect(x, mid - mx * mid * 0.92, 1, Math.max(1, (mx - mn) * mid * 0.92));
    }
    if ($(".pt").checked) {
      const f = st.item.f0, voiced = f.filter(v => v > 0);
      if (voiced.length) {
        const lo = Math.min(...voiced) - 5, hi = Math.max(...voiced) + 5;
        g.strokeStyle = css2("--pitch"); g.lineWidth = 1.6; g.beginPath(); let on = false;
        for (let x = 0; x < W; x++) { const v = f[Math.floor(x / W * f.length)];
          if (v > 0) { const yy = H - 6 - (v - lo) / (hi - lo) * (H - 12); on ? g.lineTo(x, yy) : g.moveTo(x, yy); on = true; } else on = false; }
        g.stroke();
      }
    }
    g.fillStyle = css2("--ink"); g.fillRect(px, 0, 1.5, H);
    g.fillStyle = css2("--mute"); g.font = "10px system-ui";
    for (let s = 0; s <= Math.floor(dur); s++) g.fillText(s + "s", Math.min(W - 16, s / dur * W + 2), H - 3);
  }

  function frame() {
    if (!st.playing) return;
    const p = pos(); $(".pos").textContent = p.toFixed(3); draw();
    if (p >= st.buf.duration) { stop(false); st.offset = 0; $(".pos").textContent = "0.000"; draw(); return; }
    requestAnimationFrame(frame);
  }
  function play() {
    if (!st.buf) return;
    if (active && active !== api) active.pause();
    const c = ctx(); c.resume(); const src = c.createBufferSource(); src.buffer = st.buf; src.connect(c.destination);
    src.start(0, st.offset); st.src = src; st.t0 = c.currentTime - st.offset; st.playing = true; active = api;
    $(".play").innerHTML = "&#10074;&#10074;"; frame();
  }
  function pause() { if (st.playing) { st.offset = pos(); stop(false); } }
  function stop(reset) {
    if (st.src) { try { st.src.stop(); } catch (e) {} st.src = null; }
    st.playing = false; $(".play").innerHTML = "&#9654;"; if (reset) st.offset = 0; draw();
  }
  const api = { pause };

  function move(dir) {
    const sel = st.sel, i = sel.indexOf(st.n);
    if (i < 0) { const nx = dir > 0 ? sel.find(v => v > st.n) : [...sel].reverse().find(v => v < st.n); if (nx) load(nx); return; }
    load(sel[Math.min(sel.length - 1, Math.max(0, i + dir))]);
  }
  function parse(txt) {
    const out = new Set();
    for (const part of txt.split(",")) { const m = part.trim().match(/^(\d+)\s*(?:-\s*(\d+))?$/); if (!m) continue;
      const a = +m[1], b = m[2] ? +m[2] : a; for (let i = Math.min(a, b); i <= Math.max(a, b); i++) if (i >= 1 && i <= 114) out.add(i); }
    return [...out].sort((a, b) => a - b);
  }
  $(".go").onclick = () => { const v = parseInt($(".idbox").value, 10); if (v >= 1 && v <= 114) load(v); };
  $(".idbox").addEventListener("keydown", e => { if (e.key === "Enter") $(".go").click(); });
  $(".prev").onclick = () => move(-1); $(".next").onclick = () => move(1);
  $(".apply").onclick = () => { const s = parse($(".selbox").value); if (s.length) { st.sel = s; load(s[0]); } };
  $(".fl").onclick = () => { if (flagged.length) { st.sel = flagged.slice(); $(".selbox").value = flagged.join(", "); load(flagged[0]); } };
  $(".all").onclick = () => { st.sel = ALL.slice(); $(".selbox").value = ""; load(st.n); };
  $(".play").onclick = () => st.playing ? pause() : play();
  $(".pt").onchange = draw;
  cv.onclick = e => { if (!st.buf) return; const r = cv.getBoundingClientRect(), was = st.playing; if (was) stop(false);
    st.offset = Math.max(0, Math.min(st.buf.duration - 0.01, (e.clientX - r.left) / r.width * st.buf.duration));
    $(".pos").textContent = st.offset.toFixed(3); draw(); if (was) play(); };
  window.addEventListener("resize", draw);
  load(1);
}
build("pidgin"); build("camfranglais");
})();
</script>
"""

display(HTML(_A9_HTML.replace("__FLAGGED__", json.dumps(_A9_FLAGGED))))

save_code("01_A9_listen_flagged", "Manual audio audit: listen to every clip next to its own sentence, with a moving waveform; Pidgin and Camfranglais controlled separately")