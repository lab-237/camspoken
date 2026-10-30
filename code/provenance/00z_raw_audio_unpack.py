# 00z_raw_audio_unpack
# AIM: Optional, provenance only: unpack the raw recordings (not released) so that 01_A5_trim_audio can be rerun
# Run after 00a, only if you have the raw recordings as two zip files in the inputs/ folder (one with 'pidgin', one with 'camf' in its name).

# Guard: stops with a clear message if 00a has not been run in this session.
assert "INPUT_DIR" in globals(), "Run 00a_session_and_data.py first, in the same session."

AUDIO_EXT = {".m4a", ".wav", ".mp3", ".aac", ".flac", ".ogg", ".mp4"}
def extract_zip(zip_path, out):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
def list_recordings(folder):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT
            and "__MACOSX" not in p.parts and not p.name.startswith(".")]
def item_number(p):
    return int(re.findall(r"\d+", p.stem)[-1])
def find_zip(keyword):
    hits = [p for p in INPUT_DIR.glob("*.zip") if keyword in p.name.lower()]
    assert len(hits) == 1, f"expected one zip containing '{keyword}', found {hits}"
    return hits[0]

sources = {}
for variety, kw in (("pidgin", "pidgin"), ("camfranglais", "camf")):
    out = Path("/content/raw_audio") / variety
    extract_zip(find_zip(kw), out)
    recs = sorted(list_recordings(out), key=item_number)
    assert sorted(item_number(p) for p in recs) == list(range(1, 115)), f"{variety}: numbering is not 1..114"
    sources[variety] = recs
pidgin_sources, camf_sources = sources["pidgin"], sources["camfranglais"]
print(f"✅ raw audio unpacked: {len(pidgin_sources)} Pidgin, {len(camf_sources)} Camfranglais, numbering 1..114")

save_code("00z_raw_audio_unpack", "Optional, provenance only: unpack the raw recordings so that 01_A5_trim_audio can be rerun")
