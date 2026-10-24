"""Three-layer data cache discipline (owner directive, 19:40):

  layer 1  sources        - data/raw/*, data/raw/plain/* : immutable, size-verified
  layer 2  mined records  - data/cache/<name>/<shard-stem>.jsonl + manifest.json
           canonical, mined ONCE per (converter-hash, seed, source-identity);
           re-runs are skipped by the manifest, so a converter change or a new
           shard only ever (re)mines what changed
  layer 3  assembled mix  - data/mixes/<name>.jsonl + provenance manifest
           selection (seeded hash-gate over record ids), exclusion lists, and
           train/eval splits - recompute in SECONDS, re-seed for free, never
           re-mines. Gates (contamination purges) live here as exclusion files.

Determinism note: per-shard mining means vocab pools (infobox params) are
per-shard, not stream-global - records are deterministic given
(shard, seed, converter-hash), which is the invariant that matters.
"""
import hashlib, json, time
from pathlib import Path

CACHE = Path("data/cache")
MIXES = Path("data/mixes")


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def file_id(path) -> dict:
    p = Path(path)
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}


def manifest_path(out: Path) -> Path:
    return out.with_suffix(".manifest.json")


def is_current(out: Path, converter_sha: str, seed: int, source) -> bool:
    mp = manifest_path(out)
    if not out.exists() or not mp.exists():
        return False
    m = json.loads(mp.read_text())
    src = source if isinstance(source, list) else [source]
    return (m["converter_sha"] == converter_sha and m["seed"] == seed
            and [f["size"] for f in m["sources"]] == [Path(s).stat().st_size for s in src])


def write_manifest(out: Path, converter_sha: str, seed: int, sources, n_records: int):
    manifest_path(out).write_text(json.dumps({
        "converter_sha": converter_sha, "seed": seed,
        "sources": [file_id(s) for s in sources],
        "records": n_records, "produced": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))


def id_selected(rid: str, salt: str, mod: int, lt: int) -> bool:
    """Stable hash-gate: same (rid, salt) -> same answer forever. Cheap re-seed."""
    h = int(hashlib.md5(f"{salt}:{rid}".encode()).hexdigest()[:8], 16)
    return h % mod < lt
