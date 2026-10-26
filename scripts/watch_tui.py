#!/usr/bin/env python3
"""clarke-flash trainer dashboard (ncurses).

    ./scripts/watch_training.sh            # TTY: this dashboard; non-TTY: old painter
    ./scripts/watch_training.sh 2          # 2s refresh
    ./scripts/watch_training.sh status     # one plain-text frame (logs/cron)
    python3 scripts/watch_tui.py --once    # same, directly

Shows ONE run: the active one, or the most recent when nothing is training.
Panels: current run + component losses, GPU, CPU (per core), memory/swap/disk,
network (interfaces that actually hold an IP), log tail.

Keys: q quit | space pause | l cycle log | r reset samplers | ? help
"""
from __future__ import annotations

import curses
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = Path(os.environ.get("LOGDIR", "/tmp/opencode"))
HIST = 60
SPARK = "▁▂▃▄▅▆▇█"
EIGHTHS = "▏▎▍▌▋▊▉█"
COMPS = ("null", "span", "cand", "op", "present", "dist")

C_ACCENT, C_GOOD, C_WARN, C_BAD, C_DIM = 1, 2, 3, 4, 5

# parsed-run cache: {(path, mtime): extra} - avoids re-scanning a 1MB jsonl every second
_CACHE: dict[tuple[str, float], dict] = {}


# ----------------------------------------------------------------- data sources
def _procs() -> list[str]:
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:
            continue
        if "clarke-" in raw:
            out.append(raw)
    return out


def _scan_run(path: Path) -> dict:
    """Last train row, last val row, best val and checkpoint inventory (cached by mtime)."""
    key = (str(path), os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]
    last_train, prev_train = None, None
    last_val, prev_val, best_val, best_step = None, None, None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("val"):
                    prev_val, last_val = last_val, row
                    if isinstance(row.get("loss"), (int, float)):
                        if best_val is None or row["loss"] < best_val:
                            best_val, best_step = row["loss"], row.get("step")
                else:
                    prev_train, last_train = last_train, row
    except OSError:
        return {}
    ckpts = sorted(int(p.rsplit("checkpoint-", 1)[1])
                   for p in glob.glob(str(path.parent / "checkpoint-*"))
                   if p.rsplit("checkpoint-", 1)[-1].isdigit())
    info = {"train": last_train, "prev_train": prev_train, "val": last_val,
            "prev_val": prev_val, "best_val": best_val,
            "best_step": best_step, "ckpts": ckpts}
    _CACHE[key] = info
    return info


def current_run() -> dict | None:
    """The active run if one is training, else the most recently touched run."""
    cmds = _procs()
    best = None
    for path in glob.glob(str(ROOT / "runs" / "*" / "train.jsonl")):
        p = Path(path)
        info = _scan_run(p)
        train = info.get("train")
        if not train:
            continue
        run_dir = p.parent
        rec = {
            "name": run_dir.name,
            "step": train.get("step") or 0,
            "total": train.get("total") or 0,
            "epoch": train.get("epoch") or 0,
            "loss": train.get("loss"),
            "loss_prev": (info.get("prev_train") or {}).get("loss"),
            "lr": train.get("lr"),
            "rps": train.get("recs_per_s") or 0.0,
            "gpu_gb": train.get("gpu_gb") or 0.0,
            "bs": train.get("bs") or 1,
            "accum": train.get("accum") or 1,
            "comps": {k: train[k] for k in COMPS if isinstance(train.get(k), (int, float))},
            "best_val": info.get("best_val"),
            "best_step": info.get("best_step"),
            "val": (info.get("val") or {}).get("loss"),
            "val_step": (info.get("val") or {}).get("step"),
            "val_prev": (info.get("prev_val") or {}).get("loss"),
            "ckpts": info.get("ckpts") or [],
            "mtime": os.path.getmtime(p),
            "active": any("clarke-train" in c and (str(run_dir) in c or run_dir.name in c)
                          for c in cmds),
        }
        if rec["active"]:
            return rec
        if best is None or rec["mtime"] > best["mtime"]:
            best = rec
    return best


def gpu_stats() -> dict | None:
    try:
        line = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,"
             "power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not line:
        return None
    v = [x.strip() for x in line[0].split(",")]

    def num(i, default=None):
        try:
            return float(v[i])
        except (ValueError, IndexError):
            return default

    return {"util": num(0, 0.0), "mem": num(1, 0.0), "mem_total": num(2, 1024.0),
            "temp": num(3, 0.0), "power": num(4, 0.0), "power_limit": num(5)}


def cpu_usage(prev: dict | None) -> tuple[list[float], dict]:
    cores, cur = [], []
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu") and line[3].isdigit():
                p = [int(x) for x in line.split()[1:]]
                idle = p[3] + (p[4] if len(p) > 4 else 0)
                cur.append((sum(p), idle))
    except OSError:
        pass
    if prev and len(prev) == len(cur):
        for (t0, i0), (t1, i1) in zip(prev, cur):
            dt, di = t1 - t0, i1 - i0
            cores.append(0.0 if dt <= 0 else max(0.0, min(100.0, 100.0 * (dt - di) / dt)))
    try:
        load = [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):
        load = [0.0, 0.0, 0.0]
    return cores, {"prev": cur, "load": load}


def mem_stats() -> dict:
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = float(rest.split()[0]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return {"total": 0.0, "used": 0.0, "swap_total": 0.0, "swap_used": 0.0}
    return {"total": info.get("MemTotal", 0.0),
            "used": info.get("MemTotal", 0.0) - info.get("MemAvailable", 0.0),
            "swap_total": info.get("SwapTotal", 0.0),
            "swap_used": info.get("SwapTotal", 0.0) - info.get("SwapFree", 0.0)}


def disk_stats() -> dict:
    try:
        du = shutil.disk_usage(str(ROOT))
        return {"free": du.free / 1024 ** 3, "total": du.total / 1024 ** 3,
                "pct": 100.0 * du.used / du.total}
    except OSError:
        return {"free": 0.0, "total": 0.0, "pct": 0.0}


def ip_addrs() -> dict[str, str]:
    """Interfaces holding a routable address - a counter with no IP is not 'the network'."""
    found: dict[str, str] = {}
    try:
        out = subprocess.run(["ip", "-o", "addr", "show"], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return found
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        iface = parts[1].split("@")[0]
        if iface == "lo":
            continue
        if "scope link" in line or "scope host" in line:
            continue
        is4, is6 = " inet " in line, " inet6 " in line
        if not (is4 or is6):
            continue
        fam = "inet" if is4 else "inet6"
        try:
            addr = parts[parts.index(fam) + 1].split("/")[0]
        except (ValueError, IndexError):
            continue
        if is4:
            found[iface] = addr                      # IPv4 wins over IPv6 when both exist
        else:
            found.setdefault(iface, addr)
    return found


def net_counters() -> dict[str, tuple[int, int]]:
    cur = {}
    try:
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, _, rest = line.partition(":")
            name = name.strip()
            if name == "lo":
                continue
            v = rest.split()
            cur[name] = (int(v[0]), int(v[8]))
    except (OSError, ValueError, IndexError):
        pass
    return cur


def tail_log(path: Path, n: int) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 16384))
            return fh.read().decode("utf-8", "replace").splitlines()[-n:] if n > 0 else []
    except OSError:
        return [f"(no log at {path})"]


# ----------------------------------------------------------------- formatting
def hb(b: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if b < 1024 or unit == "T":
            return f"{b:.0f}{unit}" if unit == "B" else f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}T"


def pbar(pct: float, width: int) -> str:
    pct = max(0.0, min(100.0, pct))
    eighths = int(round(pct / 100 * width * 8))
    full, rem = divmod(eighths, 8)
    out = "█" * min(full, width)
    if full < width and rem:
        out += EIGHTHS[rem - 1]
    return out.ljust(width, "░")


def colour_for(pct: float) -> int:
    return C_BAD if pct >= 90 else (C_WARN if pct >= 75 else C_GOOD)


def spark(vals: list[float], width: int) -> str:
    if not vals:
        return "·" * width
    vals = list(vals)[-width:]
    peak = max(vals) or 1.0
    return "".join(SPARK[min(7, int(v / peak * 7))] for v in vals).rjust(width)


def eta_str(run: dict) -> str:
    if not run["total"] or run["step"] >= run["total"] or run["rps"] <= 0:
        return "—"
    secs = (run["total"] - run["step"]) / (run["rps"] / (run["bs"] * run["accum"]))
    return f"{secs / 3600:.1f}h" if secs >= 3600 else f"{secs / 60:.0f}m"


def ago(secs: float) -> str:
    if secs < 90:
        return f"{secs:.0f}s ago"
    if secs < 5400:
        return f"{secs / 60:.0f}m ago"
    return f"{secs / 3600:.1f}h ago"


# ----------------------------------------------------------------- dashboard
class Dash:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.paused = False
        self.cpu_prev = None
        self.net_prev = None
        self.net_t = time.time()
        self.h_cpu: deque[float] = deque(maxlen=HIST)
        self.h_gpu: deque[float] = deque(maxlen=HIST)
        self.h_net: deque[float] = deque(maxlen=HIST)
        self.ips: dict[str, str] = {}
        self.ips_t = 0.0
        self.log_idx = 0

    # -- primitives
    def put(self, win, y: int, x: int, text: str, attr: int = 0) -> None:
        """Clip by CHARACTERS: box-drawing glyphs are 3 UTF-8 bytes and addnstr() would
        count bytes, silently truncating frames to a third of their width."""
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x > w - 1:
            return
        text = text[: w - x]
        if not text:
            return
        try:
            win.addstr(y, x, text, attr)
        except curses.error:
            try:
                win.addstr(y, x, text[:-1], attr)   # last cell can refuse; drop one glyph
            except curses.error:
                pass

    def hline(self, win, y: int, x: int, w: int, ch: str = "─") -> None:
        self.put(win, y, x, ch * max(0, w), curses.color_pair(C_DIM))

    def box(self, win, y: int, x: int, h: int, w: int, title: str = "") -> bool:
        if h < 3 or w < 6:
            return False
        self.put(win, y, x, "╭" + "─" * (w - 2) + "╮", curses.color_pair(C_DIM))
        for i in range(1, h - 1):
            self.put(win, y + i, x, "│", curses.color_pair(C_DIM))
            self.put(win, y + i, x + w - 1, "│", curses.color_pair(C_DIM))
        self.put(win, y + h - 1, x, "╰" + "─" * (w - 2) + "╯", curses.color_pair(C_DIM))
        if title:
            self.put(win, y, x + 2, f"┤ {title} ├", curses.color_pair(C_ACCENT))
        return True

    def gauge(self, win, y: int, x: int, label: str, pct: float, width: int,
              right: str = "") -> None:
        self.put(win, y, x, f"{label:<8}", curses.color_pair(C_DIM))
        self.put(win, y, x + 8, pbar(pct, width), curses.color_pair(colour_for(pct)))
        tail = f" {pct:5.1f}%"
        self.put(win, y, x + 8 + width, tail, curses.color_pair(colour_for(pct)))
        if right:
            self.put(win, y, x + 8 + width + len(tail) + 2, right, curses.color_pair(C_DIM))

    def labelled(self, win, y: int, x: int, label: str, value: str, attr: int = 0) -> None:
        self.put(win, y, x, f"{label:<9}", curses.color_pair(C_DIM))
        self.put(win, y, x + 9, value, attr)

    # -- frame
    def frame(self, win) -> None:
        win.erase()
        h, w = win.getmaxyx()
        now = time.time()

        run = current_run()
        gpu = gpu_stats()
        cores, cpust = cpu_usage(self.cpu_prev)
        self.cpu_prev = cpust["prev"]
        load = cpust["load"]
        mem = mem_stats()
        disk = disk_stats()
        if now - self.ips_t > 5:
            self.ips, self.ips_t = ip_addrs(), now
        counters = net_counters()
        dt = max(1e-3, now - self.net_t)
        rates = {n: (max(0.0, (rx - self.net_prev[n][0]) / dt),
                     max(0.0, (tx - self.net_prev[n][1]) / dt))
                 for n, (rx, tx) in counters.items() if self.net_prev and n in self.net_prev}
        self.net_prev, self.net_t = counters, now
        # only interfaces that hold an IP; fall back to all if `ip` is unavailable
        shown = {n: r for n, r in rates.items() if not self.ips or n in self.ips}
        if cores:
            self.h_cpu.append(sum(cores) / len(cores))
        if gpu:
            self.h_gpu.append(gpu["util"])
        if shown:
            self.h_net.append(sum(r + t for r, t in shown.values()))

        y = 0
        active = bool(run and run["active"])
        state = "ACTIVE" if active else ("IDLE" if run else "NO RUN")
        self.put(win, y, 1, "clarke-flash", curses.color_pair(C_ACCENT) | curses.A_BOLD)
        self.put(win, y, 14, time.strftime("%H:%M:%S"), curses.A_BOLD)
        self.put(win, y, w - len(state) - 3, f"● {state}",
                 curses.color_pair(C_GOOD if active else C_WARN) | curses.A_BOLD)
        y += 1

        if run:
            pct = 100.0 * run["step"] / run["total"] if run["total"] else 0.0
            self.put(win, y, 1, "run", curses.color_pair(C_DIM))
            self.put(win, y, 7, run["name"][:28], curses.A_BOLD)
            self.put(win, y, max(38, w // 2),
                     f"step {run['step']:,} / {run['total']:,}", curses.color_pair(C_DIM))
            y += 1

            bw = max(20, w - 26)
            self.put(win, y, 7, pbar(pct, bw), curses.color_pair(colour_for(pct)))
            self.put(win, y, 8 + bw, f" {pct:5.1f}%", curses.color_pair(colour_for(pct)))
            y += 1

            loss = run["loss"] or 0.0
            prev_loss = run.get("loss_prev")
            arrow = ""
            if isinstance(prev_loss, (int, float)):
                delta = loss - prev_loss
                arrow = " ↓" if delta < -0.05 else (" ↑" if delta > 0.05 else " ·")
            lattr = curses.color_pair(C_GOOD if arrow.startswith(" ↓") else
                                      (C_WARN if arrow.startswith(" ↑") else C_ACCENT))
            head = f"loss {loss:.3f}{arrow}"
            self.put(win, y, 7, head, lattr)
            self.put(win, y, 7 + len(head) + 1,
                     f"lr {(run['lr'] or 0):.2e}   {run['rps']:5.1f} rec/s   "
                     f"ep {run['epoch']}   eta {eta_str(run)}", curses.color_pair(C_DIM))
            y += 1
        self.hline(win, y, 0, w)
        y += 1

        # ── GPU | CPU ────────────────────────────────────────────────────
        half = (w - 3) // 2
        right_x, right_w = half + 3, w - half - 3
        cell = 4
        per_row = max(1, (right_w - 4) // cell)
        core_rows = max(1, (min(32, os.cpu_count() or 8) + per_row - 1) // per_row)
        mid_h = max(6, min(3 + core_rows + 1, 9))

        self.box(win, y, 0, mid_h, half + 1, "GPU")
        gw = max(8, half - 34)
        if gpu:
            self.gauge(win, y + 1, 2, "util", gpu["util"], gw, "busy")
            vp = 100.0 * gpu["mem"] / max(gpu["mem_total"], 1)
            self.gauge(win, y + 2, 2, "vram", vp, gw,
                       f"{gpu['mem'] / 1024:.1f}/{gpu['mem_total'] / 1024:.0f} GiB")
            pw = (f"{gpu['power']:.0f}/{gpu['power_limit']:.0f} W"
                  if gpu.get("power_limit") else f"{gpu['power']:.0f} W")
            self.put(win, y + 3, 2, f"{'temp':<8}", curses.color_pair(C_DIM))
            self.put(win, y + 3, 10, f"{gpu['temp']:.0f}°C",
                     curses.color_pair(C_BAD if gpu["temp"] >= 85 else
                                       (C_WARN if gpu["temp"] >= 75 else C_GOOD)))
            self.put(win, y + 3, max(20, half - 20), pw, curses.color_pair(C_DIM))
            if mid_h >= 6:
                self.put(win, y + mid_h - 2, 2,
                         "util " + spark(list(self.h_gpu), max(8, half - 9)),
                         curses.color_pair(C_ACCENT))
        else:
            self.put(win, y + 1, 2, "nvidia-smi unavailable", curses.color_pair(C_BAD))

        self.box(win, y, right_x, mid_h, right_w, "CPU")
        if cores:
            self.gauge(win, y + 1, right_x + 2, "total", sum(cores) / len(cores),
                       max(8, right_w - 32), f"load {load[0]:.2f}")
            for i, u in enumerate(cores[:32]):
                ry = y + 2 + i // per_row
                if ry >= y + mid_h - 1:
                    break
                self.put(win, ry, right_x + 2 + (i % per_row) * cell, pbar(u, 3),
                         curses.color_pair(colour_for(u)))
            if mid_h >= 3 + core_rows + 1:
                self.put(win, y + mid_h - 2, right_x + 2,
                         "cpu " + spark(list(self.h_cpu), max(6, right_w - 8)),
                         curses.color_pair(C_ACCENT))
        else:
            self.put(win, y + 1, right_x + 2, "sampling…", curses.color_pair(C_DIM))
        y += mid_h

        # ── MEMORY | NETWORK ─────────────────────────────────────────────
        low_h = 6
        self.box(win, y, 0, low_h, half + 1, "MEMORY")
        mw = max(8, half - 34)
        self.gauge(win, y + 1, 2, "mem", 100.0 * mem["used"] / max(mem["total"], 1), mw,
                   f"{mem['used']:.1f}/{mem['total']:.1f} GB")
        sp = 100.0 * mem["swap_used"] / mem["swap_total"] if mem["swap_total"] else 0.0
        self.gauge(win, y + 2, 2, "swap", sp, mw,
                   f"{mem['swap_used']:.1f}/{mem['swap_total']:.1f} GB")
        self.gauge(win, y + 3, 2, "disk", disk["pct"], mw, f"{disk['free']:.0f} GiB free")

        # wide: iface + address + rates. narrow: address moves into the title so the
        # rate columns never collide.
        narrow = right_w < 54
        only = next(iter(shown), None) if len(shown) == 1 else None
        title = f"NET {self.ips.get(only, '—')}" if (narrow and only) else "NETWORK"
        self.box(win, y, right_x, low_h, right_w, title)
        if narrow:
            self.put(win, y + 1, right_x + 2, f"{'iface':<11}{'rx':>7} {'tx':>7}",
                     curses.color_pair(C_DIM) | curses.A_BOLD)
            fmt = lambda n, rx, tx: (f"{n[:10]:<11}{hb(rx) + '/s':>7} {hb(tx) + '/s':>7}")
        else:
            self.put(win, y + 1, right_x + 2,
                     f"{'iface':<11}{'address':<17}{'rx':>8}{'tx':>8}",
                     curses.color_pair(C_DIM) | curses.A_BOLD)
            fmt = lambda n, rx, tx: (f"{n[:10]:<11}{self.ips.get(n, '—')[:16]:<17}"
                                     f"{hb(rx) + '/s':>8}{hb(tx) + '/s':>8}")
        top = sorted(shown.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[: low_h - 3]
        for i, (name, (rx, tx)) in enumerate(top):
            if narrow and only:
                fmt_row = f"{name[:10]:<11}{hb(rx) + '/s':>7} {hb(tx) + '/s':>7}"
            else:
                fmt_row = fmt(name, rx, tx)
            self.put(win, y + 2 + i, right_x + 2, fmt_row, curses.color_pair(C_GOOD))
        if not top:
            self.put(win, y + 2, right_x + 2, "no interface with an IP",
                     curses.color_pair(C_DIM))
        y += low_h

        # ── RUN DETAIL ───────────────────────────────────────────────────
        # Terminology mirrors ClarkeHead.compute_loss exactly: the total is the SUM of
        # per-head means (each head normalised by its own count), so every head carries
        # equal weight regardless of how many fields exercise it.
        legend = [
            ("null", "p_null BCE: is this field absent?", "field"),
            ("span", "start/end pointer CE over state tokens", "pointer"),
            ("present", "list-slot presence BCE", "list"),
            ("cand", "candidate-atom CE (numbers/dates/times)", "atom"),
            ("op", "5-way arithmetic operator CE", "scalar"),
            ("dist", "estimate-bin CE, neighbour smoothing", "horizon"),
        ]
        avail = h - y - 1
        if run and avail >= 15:
            det_h = 11
            if self.box(win, y, 0, det_h, w, "RUN DETAIL"):
                self.put(win, y + 1, 2, f"{'term':<9}{'value':>7}  {'per':<9}what it is",
                         curses.color_pair(C_DIM) | curses.A_BOLD)
                for i, (key, what, per) in enumerate(legend):
                    val = run["comps"].get(key)
                    attr = curses.color_pair(C_GOOD) if val is None else 0
                    self.put(win, y + 2 + i, 2, f"{key:<9}")
                    self.put(win, y + 2 + i, 11,
                             f"{'—' if val is None else f'{val:7.3f}'}", attr or 0)
                    self.put(win, y + 2 + i, 20, f"{per:<9}", curses.color_pair(C_DIM))
                    self.put(win, y + 2 + i, 30, what if val is not None
                             else f"{what}  (no gradient this row)", curses.color_pair(C_DIM))
                y2 = y + 8
                val_txt = "—" if run["val"] is None else f"{run['val']:.3f}"
                arr = ""
                if isinstance(run["val"], (int, float)) and isinstance(run["val_prev"], (int, float)):
                    d = run["val"] - run["val_prev"]
                    arr = " ↓" if d < -0.005 else (" ↑" if d > 0.005 else " ·")
                best = ("—" if run["best_val"] is None
                        else f"{run['best_val']:.3f} @ {run['best_step']}")
                n_ck = len(run["ckpts"])
                ck = f"ckpts {n_ck}" + (f" (latest {run['ckpts'][-1]})" if n_ck else "")
                self.put(win, y2, 2, "val", curses.color_pair(C_DIM))
                self.put(win, y2, 6, f"{val_txt}{arr}",
                         curses.color_pair(C_GOOD if arr.startswith(" ↓") else
                                           (C_WARN if arr.startswith(" ↑") else C_ACCENT)))
                self.put(win, y2, 18, f"best {best}    {ck}", curses.color_pair(C_DIM))
                stale = now - run["mtime"]
                self.put(win, y2 + 1, 2, "now", curses.color_pair(C_DIM))
                self.put(win, y2 + 1, 6,
                         f"gpu {run['gpu_gb']:.1f} GB   bs {run['bs']}×{run['accum']}   "
                         f"step {run['step']:,}/{run['total']:,}   ",
                         curses.color_pair(C_DIM))
                self.put(win, y2 + 1, 6 + 46, f"last log {ago(stale)}",
                         curses.color_pair(C_BAD if stale > 180 else
                                           (C_WARN if stale > 90 else C_GOOD)))
            y += det_h
        elif run:
            det_h = 4
            if self.box(win, y, 0, det_h, w, "RUN DETAIL"):
                comps = "   ".join(f"{k} {v:.3f}" for k, v in run["comps"].items())
                self.put(win, y + 1, 2, comps or "no component losses in the latest row")
                val = ("—" if run["best_val"] is None
                       else f"best val {run['best_val']:.3f} @ {run['best_step']}")
                self.put(win, y + 2, 2,
                         f"{val}   ckpts {len(run['ckpts'])}   gpu {run['gpu_gb']:.1f} GB   "
                         f"last log {ago(now - run['mtime'])}")
            y += det_h

        # ── log tail ─────────────────────────────────────────────────────
        log_h = h - y - 1
        if log_h >= 4:
            logs = sorted(glob.glob(str(LOGDIR / "*.log")), key=os.path.getmtime,
                          reverse=True)[:8]
            path = Path(logs[self.log_idx % len(logs)]) if logs else Path("-")
            title = f"LOG {path.name}"
            if len(logs) > 1:
                title += f"  ({self.log_idx % len(logs) + 1}/{len(logs)} · l)"
            if self.box(win, y, 0, log_h, w, title):
                for i, line in enumerate(tail_log(path, log_h - 2)):
                    low = line.lower()
                    attr = (curses.color_pair(C_BAD) if ("failed" in low or "error" in low)
                            else (curses.color_pair(C_GOOD)
                                  if ("done" in low or "trained" in low) else 0))
                    self.put(win, y + 1 + i, 2, line, attr or curses.color_pair(C_DIM))

        pause_txt = " · PAUSED" if self.paused else ""
        bar = (" q quit   space pause   l log   r reset   ? help "
               f"  [{self.interval:g}s{pause_txt}] ")
        self.put(win, h - 1, 0, bar.ljust(w - 1),
                 curses.color_pair(C_ACCENT) | curses.A_REVERSE)
        win.noutrefresh()
        curses.doupdate()

    def help(self, win) -> None:
        lines = ["keys", "", "q / esc   quit", "space     pause refresh",
                 "l         next log file", "r         reset samplers", "?         this help",
                 "", "shows the ACTIVE run, or the most recent when idle",
                 "network lists interfaces holding an IP only"]
        self.box(win, 2, 3, len(lines) + 4, 60, "HELP")
        for i, line in enumerate(lines):
            self.put(win, 3 + i, 5, line)
        self.put(win, len(lines) + 4, 5, "press any key", curses.color_pair(C_DIM))
        win.noutrefresh()
        curses.doupdate()
        win.nodelay(False)
        win.getch()
        win.nodelay(True)

    def run(self, stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            try:
                curses.use_default_colors()
            except curses.error:
                pass
            curses.init_pair(C_ACCENT, curses.COLOR_CYAN, -1)
            curses.init_pair(C_GOOD, curses.COLOR_GREEN, -1)
            curses.init_pair(C_WARN, curses.COLOR_YELLOW, -1)
            curses.init_pair(C_BAD, curses.COLOR_RED, -1)
            curses.init_pair(C_DIM, curses.COLOR_BLUE, -1)
        _, first = cpu_usage(None)
        self.cpu_prev = first["prev"]
        self.net_prev = net_counters()
        self.ips = ip_addrs()
        self.ips_t = time.time()
        time.sleep(0.3)
        while True:
            self.frame(stdscr)
            deadline = time.time() + (self.interval * 4 if self.paused else self.interval)
            while time.time() < deadline:
                ch = stdscr.getch()
                if ch == -1:
                    time.sleep(0.05)
                    continue
                if ch in (ord("q"), 27):
                    return
                if ch == ord(" "):
                    self.paused = not self.paused
                    break
                if ch == ord("l"):
                    self.log_idx += 1
                    break
                if ch == ord("r"):
                    self.cpu_prev = None
                    break
                if ch == ord("?"):
                    self.help(stdscr)
                    break


def plain_frame() -> str:
    run = current_run()
    gpu = gpu_stats()
    mem = mem_stats()
    _, first = cpu_usage(None)
    time.sleep(0.2)
    cores, _ = cpu_usage(first["prev"])
    lines = [f"clarke-flash {time.strftime('%H:%M:%S')}"]
    if run:
        tag = "ACTIVE" if run["active"] else "last"
        lines.append(f"{run['name']:<16}{run['step']:>6}/{run['total']:<6} ep{run['epoch']} "
                     f"loss {run['loss']:.3f}  {run['rps']:5.1f} r/s  "
                     f"{run['gpu_gb']:4.1f}G  {tag}")
        comps = "  ".join(f"{k} {v:.3f}" for k, v in run["comps"].items())
        if comps:
            lines.append(comps)
    else:
        lines.append("no run yet")
    if gpu:
        lines.append(f"gpu  util {gpu['util']:.0f}%  "
                     f"vram {gpu['mem'] / 1024:.1f}/{gpu['mem_total'] / 1024:.0f}G  "
                     f"{gpu['temp']:.0f}C  {gpu['power']:.0f}W")
    else:
        lines.append("gpu  nvidia-smi unavailable")
    cpu_txt = f"  cpu {sum(cores) / len(cores):.0f}%  cores {os.cpu_count()}" if cores else ""
    lines.append(f"mem  {mem['used']:.1f}/{mem['total']:.1f}G  "
                 f"swap {mem['swap_used']:.1f}/{mem['swap_total']:.1f}G{cpu_txt}")
    ips = ip_addrs()
    if ips:
        lines.append("net  " + "  ".join(f"{n} {a}" for n, a in ips.items()))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    interval, once = 1.0, "--once" in argv
    for a in argv:
        try:
            interval = max(0.2, float(a))
        except ValueError:
            continue
    if once or not sys.stdout.isatty():
        print(plain_frame())
        return 0
    curses.wrapper(Dash(interval).run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
