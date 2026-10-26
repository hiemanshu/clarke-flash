"""Aggregate per-shard census JSONs into one corpus report."""
import json, glob, sys
from collections import Counter, defaultdict

shards = [json.load(open(f)) for f in sorted(glob.glob("reports/census/*.json"))]
if not shards:
    sys.exit("no census results yet")
tot = sum(s["pages"] for s in shards)
print(f"shards censused: {len(shards)}   pages: {tot:,}")
print(f"\n{'family':24s} {'coverage':>9s}  {'mean/ page':>10s}")
fams = ["infobox", "shortdesc", "categories", "anytable", "cites", "external_links",
        "sections", "files_images", "life_date_templates", "ipa_scripts", "flags",
        "hatnote", "succession", "disambiguation_page", "redirect_page"]
for f in fams:
    w = sum(s["coverage_pct"].get(f, 0) * s["pages"] for s in shards) / tot
    print(f"  {f:22s} {w:8.1f}%")
print(f"\n{'mean counts':24s}")
keys = ["categories_n", "sections_n", "wikitables_n", "cites_n", "images_n", "bullets_n", "infobox_params_n"]
for k in keys:
    w = sum(s["mean_counts"].get(k, 0) * s["pages"] for s in shards) / tot
    print(f"  {k:22s} {w:8.2f}")
types = Counter()
for s in shards:
    types.update(dict(s["top_infobox_types"]))
print("\ntop infobox types (across censused shards):")
for t, n in types.most_common(12):
    print(f"  {t:24s} {n:,}")
print("\nper-shard:")
print(f"  {'shard':46s} {'pages':>9s} {'redir':>6s} {'ib':>6s} {'sd':>6s} {'tbl':>6s} {'cite':>6s}")
for s in shards:
    c = s["coverage_pct"]; nm = s["shard"].replace(".xml.bz2", "")
    print(f"  {nm:46s} {s['pages']:9,} {c.get('redirect_page',0):5.1f}% {c.get('infobox',0):5.1f}% {c.get('shortdesc',0):5.1f}% {c.get('anytable',0):5.1f}% {c.get('cites',0):5.1f}%")
