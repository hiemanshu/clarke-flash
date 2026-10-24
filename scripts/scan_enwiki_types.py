"""enwiki XML dump -> full census of human-curated data types (inventory, not records).

Answers "what kinds of extracted-by-hand data are in here and how much of it":
per family, page coverage %, mean counts, and preserved samples. Runs over every
shard; incremental driver can re-invoke with --done-file to skip processed ones.

usage: uv run python scripts/scan_enwiki_types.py data/raw/enwiki_*.xml.bz2 [--append OUT.jsonl]
"""
import bz2, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

SD = re.compile(r"\{\{[Ss]hort description\|")
IB = re.compile(r"\{\{\s*[Ii]nfo[Bb]ox\s+([^\s|]{1,40})")
HAT = re.compile(r"\{\{(?:about|dablink|hatnote)\|")
CAT = re.compile(r"\[\[[Cc]ategory:")
WIKITABLE = re.compile(r"\{\|[^}]*class=\"[^\"]*wikitable")
ANYTABLE = re.compile(r"\{\|")
CITE = re.compile(r"\{\{[Cc]ite [a-z]+")
FILEIMG = re.compile(r"\[\[[Ff]ile:|\[\[[Ii]mage:")
MATH = re.compile(r"<math>|\{\{[Dd]isplaystyle")
IPAX = re.compile(r"\{\{(IPAc-en|IPA2?|sc)")
DATES = re.compile(r"\{\{(Bd|Born date and age|Birth date|Death date|Use dmy dates|Use mdy dates)")
SUC = re.compile(r"\{\{Succession box")
FLAGX = re.compile(r"\{\{[Ff]lag(?:icon)?\|")
EXTLNK = re.compile(r"http[s]?://")
BULLET = re.compile(r"^\*[*# ]?[^\s]", re.M)
SECTION = re.compile(r"==([^=]+)==")
DISAMB = " (disambiguation)"


def parse_pages(path):
    import io
    opener = (lambda p: open(p, encoding="utf-8", errors="replace")) if str(path).endswith(".xml") else (lambda p: io.TextIOWrapper(bz2.BZ2File(p), encoding="utf-8", errors="replace"))
    with opener(path) as f:
        buf = ""
        while True:
            try:
                chunk = f.read(1 << 22)
            except EOFError:
                break
            if not chunk:
                break
            buf += chunk
            while "</page>" in buf:
                page, buf = buf.split("</page>", 1)
                if "<page>" in page:
                    yield page.split("<page>", 1)[1]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    append = None
    if "--append" in sys.argv:
        append = Path(sys.argv[sys.argv.index("--append") + 1])
        args.remove(str(append))
    if "--out" in sys.argv:
        outp = Path(sys.argv[sys.argv.index("--out") + 1])
        args = [a for a in args if a != str(outp)]
    out = []
    for path in args:
        n = 0
        cov = Counter()
        sums = Counter()
        ibtypes = Counter()
        samples = {}
        for page in parse_pages(Path(path)):
            m = re.search(r"<ns>(\d+)</ns>", page)
            if m and m.group(1) != "0":
                continue
            tx = re.search(r"<text[^>]*>(.*?)</text>", page, re.S)
            if not tx:
                continue
            t = tx.group(1)
            n += 1
            fams = {
                "infobox": IB.search(t), "shortdesc": SD.search(t), "hatnote": HAT.search(t),
                "categories": CAT.search(t), "wikitables": WIKITABLE.search(t),
                "anytable": ANYTABLE.search(t), "cites": CITE.search(t),
                "files_images": FILEIMG.search(t), "math": MATH.search(t),
                "ipa_scripts": IPAX.search(t), "life_date_templates": DATES.search(t),
                "succession": SUC.search(t), "flags": FLAGX.search(t),
                "external_links": EXTLNK.search(t), "sections": SECTION.search(t),
                "disambiguation_page": None, "redirect_page": None,
            }
            if t.lstrip().startswith(("#REDIRECT", "#redirect")):
                fams["redirect_page"] = True
            mt = re.search(r"<title>([^<]+)</title>", page)
            if mt and mt.group(1).endswith(DISAMB):
                fams["disambiguation_page"] = True
            for k, hit in fams.items():
                if hit:
                    cov[k] += 1
            counts = {
                "categories_n": len(CAT.findall(t)), "sections_n": len(SECTION.findall(t)),
                "wikitables_n": len(ANYTABLE.findall(t)), "cites_n": len(CITE.findall(t)),
                "images_n": len(FILEIMG.findall(t)), "bullets_n": len(BULLET.findall(t)),
                "infobox_params_n": len(re.findall(r"^\|[a-z]", t[:5000], re.M)),
            }
            for k, v in counts.items():
                sums[k] += v
            for m2 in IB.finditer(t):
                ibtypes[m2.group(1).strip().lower()] += 1
            for k, hit in fams.items():
                if hit and k not in samples and hasattr(hit, "group"):
                    s = max(0, hit.start() - 10)
                    samples[k] = t[s:s + 150]
        shard = {
            "shard": Path(path).name, "pages": n,
            "coverage_pct": {k: round(100 * v / max(n, 1), 1) for k, v in cov.items()},
            "mean_counts": {k: round(v / max(n, 1), 2) for k, v in sums.items()},
            "top_infobox_types": ibtypes.most_common(20),
            "samples": samples,
        }
        print(json.dumps({k: shard[k] for k in ("shard", "pages", "coverage_pct", "mean_counts")}))
        out.append(shard)
        if append:
            with append.open("a") as h:
                h.write(json.dumps(shard) + "\n")
    if not append:
        dest = Path("reports/enwiki_types.json") if "--out" not in sys.argv else Path(sys.argv[sys.argv.index("--out") + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out[-1] if "--out" in sys.argv and len(out) == 1 else out, indent=1))
        print(f"-> {dest}")


if __name__ == "__main__":
    main()
