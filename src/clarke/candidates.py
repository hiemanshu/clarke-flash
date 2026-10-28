"""Candidate pools: values that literally occur in the state (or combine in
one arithmetic step), with their character spans.

Clarke's contract is that numbers/dates are *read from the input* (typed
spans) or computed from it in one or two steps. So instead of regressing
digit bins, the head picks from these pools and code renders the exact value
from the span text — grounding comes for free, and arithmetic stays at the
1-2 step ceiling the blog itself describes.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from .schema import FieldSpec

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:ril)?|May|Jun(?:e)?|Jul(?:y)?|"
         r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE_RES = (
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
    re.compile(r"\d{1,2}/\d{1,2}/\d{2}(?!\d)"),
    re.compile(rf"{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"),
    re.compile(rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH}\.?,?\s+\d{{4}}"),
)
DT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?")
TIME_RE = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?")

MAX_ATOMS = 64

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%Y/%m/%d",
                "%b %d %Y", "%b %d, %Y", "%B %d %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y")

UNIT_SECONDS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400,
                "weeks": 604800, "months": 2592000, "years": 31536000}

OP_IDS = {"id": 0, "add": 1, "sub": 2, "mul": 3, "div": 4}


def parse_date(text: str) -> str | None:
    cleaned = re.sub(r"(\d)(?:st|nd|rd|th)", r"\1", text).replace("Sept", "Sep")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_time(text: str) -> str | None:
    cleaned = re.sub(r"[.\s]+", "", text).upper()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%SP", "%I:%MP"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    return None


def parse_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace("+", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _csv_aggregates(state: str, cap: int = 48) -> list[tuple[float, tuple[int, int]]]:
    """Detect CSV blocks (header + >=3 rows with matching column counts) and
    emit per-numeric-column aggregates anchored at the header cell span."""
    out: list[tuple[float, tuple[int, int]]] = []
    lines: list[tuple[int, str]] = []
    pos = 0
    for line in state.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    i = 0
    while i < len(lines) and len(out) < cap:
        base, header = lines[i]
        cells = header.split(",")
        ncols = len(cells)
        j = i + 1
        rows: list[list[str]] = []
        while j < len(lines) and len(rows) < 4096:
            rc = lines[j][1].split(",")
            if len(rc) != ncols:
                break
            rows.append(rc)
            j += 1
        if ncols >= 3 and len(rows) >= 3 and not any(
                parse_number(c.strip().strip('"')) is not None for c in cells):
            off = 0
            for ci in range(ncols):
                cell = cells[ci]
                name = cell.strip()
                cstart = off + (len(cell) - len(cell.lstrip()))
                cend = off + len(cell.rstrip())
                off += len(cell) + 1
                if len(name) < 2:
                    continue
                vals: list[float] = []
                ok = True
                for rc in rows:
                    v = parse_number(rc[ci].strip().strip('"'))
                    if v is None:
                        ok = False
                        break
                    vals.append(v)
                if not ok or len(set(vals)) < 2:
                    continue
                n = len(vals)
                s = sum(vals)
                sv = sorted(vals)
                med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
                anchor = (base + cstart, base + cend)
                out.extend([(s, anchor), (s / n, anchor), (sv[0], anchor),
                            (sv[-1], anchor), (med, anchor), (float(n), anchor)])
                if len(out) >= cap:
                    break
            i = j if rows else i + 1
        elif rows and ncols < 3:
            # A run with fewer than 3 columns can never be a header block, and every
            # line inside it has the same column count - re-examining them one by one
            # is pure waste (240k-char loan tape: 56551 lines, mostly 0 commas).
            i = j
        elif rows:
            # A rejected block must not be re-scanned from the next line: on a loan
            # tape whose "header" line is itself numeric, advancing by one re-read the
            # same multi-thousand-row run each time - O(rows^2), 220s for one document
            # (data/ledgar_split_cuad.jsonl[10313], 2026-10-27). Rows that look numeric
            # can never become a valid header, so skip to the first line in the run
            # that does not start with a number (or past the run entirely).
            skip = next((t for t, rc in enumerate(rows)
                         if parse_number(rc[0].strip().strip('"')) is None), len(rows))
            i += 1 + skip
        else:
            i += 1
    return out[:cap]


# ---------------------------------------------------------------- text candidates
# Party-like name mentions: quoted defined terms, capitalised runs, and the operands of
# "by and between X and Y". Measured on doc40: 191/192 CUAD party items (99.5%) and 39/40
# whole documents are covered by these rules, so the pointer does not have to invent the
# span - it only has to RANK ~221 candidates down to the 3-4 real parties.
_QUOTED = re.compile(r"[\"\u201c\u201d']([A-Z][^\"\u201c\u201d']{0,60})[\"\u201c\u201d']")
_CAP_RUN = re.compile(r"\b(?:[A-Z][A-Za-z&.\-]*)(?:\s+(?:[A-Z][A-Za-z&.\-]*|of|the|and|&))*")
_INTRO = re.compile(r"(?:by and between|between)\s+([^.;]{5,200})", re.IGNORECASE)


def text_candidates(state: str) -> list[tuple[str, tuple[int, int]]]:
    """(value, char span) candidates for name-like fields such as `parties`."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[str, tuple[int, int]]] = []

    suffix = re.compile(
        r"\s*,?\s*(?:Inc|LLC|L\.L\.C|Ltd|Corp|Corporation|Company|Co|S\.A|L\.P|PLC|N\.V|"
        r"GmbH|Limited|Partners|Holdings)\.?", re.IGNORECASE)

    def add(start: int, end: int) -> None:
        # Legal suffixes sit after a COMMA ("YOU ON DEMAND HOLDINGS, INC."), which the
        # capitalised-run rule stops at. Without this the crop is right but the boundary is
        # wrong, and exact-match scores it as a miss (crop-exact fell .84 -> .69 in the first
        # candidate-decode measurement). Absorb the suffix, then trim trailing punctuation.
        tail = state[end:end + 40]
        m = suffix.match(tail)
        if m:
            end = end + m.end()
        text = " ".join(state[start:end].split()).strip("\"\u201c\u201d' ,")
        if not (2 <= len(text) <= 90):
            return
        key = (start, end)
        if key in seen:
            return
        seen.add(key)
        out.append((text, (start, end)))

    for m in _QUOTED.finditer(state):
        add(m.start(1), m.end(1))
    for m in _CAP_RUN.finditer(state):
        add(m.start(), m.end())
    for m in _INTRO.finditer(state):
        chunk = m.group(1)
        cursor = m.start(1)
        for part in re.split(r"(,|\band\b|&)", chunk):
            if part not in (",", " and ", "&") or part is None:
                pass
            if part and part not in (",", "&") and part.strip().lower() != "and":
                stripped = part.strip()
                if stripped:
                    off = chunk.find(part, max(0, cursor - m.start(1)))
                    if off >= 0:
                        add(m.start(1) + off, m.start(1) + off + len(part))
            cursor += len(part or "")
    return out


class Pool:
    """Candidate atoms for one record: (value, char_span) per extracted form.

    agg_atoms=True adds VIRTUAL atoms: whole-column aggregates over CSV blocks
    (sum/mean/min/max/median/count), anchored at the column's header cell.
    They turn the table-domain wall (gold aggregates absent from the pool ->
    solvable_rate 25%) into the same pick-don't-compute problem every other
    number field already is; the +/-8-token context pooling reads the column
    name, so "total revenue" binds to the 'revenue' header by meaning.
    """

    MAX_AGG = 48

    def __init__(self, state: str, agg_atoms: bool = False):
        self.state = state
        self.numbers: list[tuple[float, tuple[int, int]]] = []
        self.aggs: list[tuple[float, tuple[int, int]]] = []
        self.dates: list[tuple[str, tuple[int, int]]] = []
        self.times: list[tuple[str, tuple[int, int]]] = []
        self.datetimes: list[tuple[str, tuple[int, int]]] = []
        self._extract(state)
        if agg_atoms:
            self.aggs = _csv_aggregates(state, self.MAX_AGG)

    def _extract(self, state: str) -> None:
        taken_iso: list[tuple[int, int]] = []
        for match in DT_RE.finditer(state):
            raw = match.group()
            try:
                canon = datetime.fromisoformat(raw if len(raw) > 16 else raw + ":00")
            except ValueError:
                continue
            self.datetimes.append((canon.strftime("%Y-%m-%dT%H:%M:%S"),
                                   (match.start(), match.end())))
            taken_iso.append((match.start(), match.end()))
        for regex in DATE_RES:
            for match in regex.finditer(state):
                iso = parse_date(match.group())
                if iso is None or any(s <= match.start() < e for s, e in taken_iso):
                    continue
                self.dates.append((iso, (match.start(), match.end())))
        for match in TIME_RE.finditer(state):
            if any(s <= match.start() < e for s, e in taken_iso):
                continue
            hms = parse_time(match.group())
            if hms:
                self.times.append((hms, (match.start(), match.end())))
        for match in NUMBER_RE.finditer(state):
            value = parse_number(match.group())
            if value is not None and len(self.numbers) < MAX_ATOMS:
                self.numbers.append((value, (match.start(), match.end())))
        self.dates = self.dates[:MAX_ATOMS]
        self.times = self.times[:MAX_ATOMS]
        self.datetimes = self.datetimes[:MAX_ATOMS]

    def duration_atoms(self, spec: FieldSpec) -> list[tuple[float, tuple[int, int]]]:
        """Candidate durations: differences between date/time atoms, in the
        declared unit, plus number atoms that are directly tagged by the unit
        word in the text ("11 days late"). Bare numbers are NOT duration
        candidates — otherwise every ticket ID pollutes the pool."""
        factor = UNIT_SECONDS.get(str(spec.unit or "days"), 86400)
        unit = str(spec.unit or "days").lower()
        out: list[tuple[float, tuple[int, int]]] = []
        moments: list[tuple[Any, tuple[int, int]]] = []
        for iso, span in self.dates:
            moments.append((datetime.fromisoformat(iso), span))
        for iso, span in self.datetimes:
            text = iso if len(iso) > 16 else iso + ":00"
            moments.append((datetime.fromisoformat(text), span))
        for i in range(len(moments)):
            for j in range(i + 1, len(moments)):
                delta = abs((moments[j][0] - moments[i][0]).total_seconds()) / factor
                span = (min(moments[i][1][0], moments[j][1][0]), max(moments[i][1][1], moments[j][1][1]))
                out.append((delta, span))
        for value, span in self.numbers:
            tail = self.state[span[1]:span[1] + 14].lstrip().lower()
            if tail.startswith(unit.rstrip("s")) or tail.startswith(unit):
                out.append((value, span))
        return out[:MAX_ATOMS * 2]


def value_features(value: float, span_chars: int, is_agg: bool = False) -> list[float]:
    import math
    return [
        1.0 if value >= 0 else 0.0,
        math.log10(abs(value) + 1.0) / 10.0,
        (abs(value) % 1) ,
        4.0 if is_agg else span_chars / 64.0,  # 4.0: virtual aggregate marker
    ]


def solve(spec: FieldSpec, pool: Pool, gold: float, decimals: int):
    """Find (a, b, op) producing the gold value, or None when unrepresentable."""
    tolerance = 1.5 * (10 ** -decimals)
    atoms = pool.duration_atoms(spec) if spec.type == "duration" else pool.numbers + pool.aggs
    for i, (value, _) in enumerate(atoms):
        if abs(value - gold) <= tolerance:
            return i, -1, OP_IDS["id"]
    for i, (a, _) in enumerate(atoms):
        for j, (b, _) in enumerate(atoms):
            if i == j:
                continue
            if abs(a + b - gold) <= tolerance:
                return i, j, OP_IDS["add"]
            if abs(a - b - gold) <= tolerance:
                return i, j, OP_IDS["sub"]
            if abs(a * b - gold) <= tolerance:
                return i, j, OP_IDS["mul"]
            if b != 0 and abs(a / b - gold) <= tolerance:
                return i, j, OP_IDS["div"]
    return None


def apply_op(a: float, b: float | None, op: int) -> float:
    if b is None or op == OP_IDS["id"]:
        return a
    if op == OP_IDS["add"]:
        return a + b
    if op == OP_IDS["sub"]:
        return a - b
    if op == OP_IDS["mul"]:
        return a * b
    return a / b if b else a


def _date_feats(iso: str) -> list[float]:
    d = date.fromisoformat(iso)
    return [(d.year - 2000) / 50.0, d.timetuple().tm_yday / 365.0, d.day / 31.0, 0.0]


def _time_feats(hms: str) -> list[float]:
    parts = [int(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return [parts[0] / 24.0, parts[1] / 60.0, parts[2] / 60.0, 0.0]


def field_atoms(spec: FieldSpec, pool: Pool) -> list[tuple[Any, tuple[int, int] | None, list[float]]]:
    """The candidate list for one field: (value, evidence span, features).

    Order is deterministic (pool order) so training targets and decoding agree.
    """
    if spec.type == "number" or (spec.type == "list" and spec.items == "number"):
        # order MUST mirror solve()'s atom list: pool.numbers then pool.aggs
        return ([(v, span, value_features(v, span[1] - span[0])) for v, span in pool.numbers]
                + [(v, span, value_features(v, span[1] - span[0], is_agg=True))
                   for v, span in pool.aggs])
    if spec.type == "duration":
        return [(v, span, value_features(v, span[1] - span[0])) for v, span in pool.duration_atoms(spec)]
    if spec.type == "date" or (spec.type == "list" and spec.items == "date"):
        return [(iso, span, _date_feats(iso)) for iso, span in pool.dates]
    if spec.type == "time" or (spec.type == "list" and spec.items == "time"):
        return [(hms, span, _time_feats(hms)) for hms, span in pool.times]
    if spec.type == "datetime":
        return [(iso, span,
                 [_date_feats(iso[:10])[0]] + _time_feats(iso[11:])[:3])
                for iso, span in pool.datetimes]
    return []
