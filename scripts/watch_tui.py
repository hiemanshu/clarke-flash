#!/usr/bin/env python3
"""clarke-flash trainer dashboard (ncurses).

    ./scripts/watch_training.sh            # TTY: this dashboard; non-TTY: old painter
    ./scripts/watch_training.sh 2          # 2s refresh
    ./scripts/watch_training.sh status     # one plain-text frame (logs/cron)
    python3 scripts/watch_tui.py --once    # same, directly

Panels: active run, GPU (util/VRAM/temp/power), CPU (total + per core + load),
memory/swap, network (per interface with sparkline), all runs, log tail.

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

# colour pairs
C_ACCENT, C_GOOD, C_WARN, C_BAD, C_DIM = 1, 2, 3, 4, 5


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


def _last_train_rows(path: Path, want: int = 2) -> list[dict]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 8192))
            rows: list[dict] = []
            for line in reversed(fh.read().decode("utf-8", "replace").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("val"):
                    rows.append(row)
                    if len(rows) >= want:
                        break
    except OSError:
        return rows
    return rows


def discover_runs() -> list[dict]:
    """Auto-discovered runs (never a hardcoded name: new experiments must appear)."""
    cmds = _procs()
    runs = []
    for path in glob.glob(str(ROOT / "runs" / "*" / "train.jsonl")):
        run_dir = Path(path).parent
        recent = _last_train_rows(Path(path))
        if not recent:
            continue
        last = recent[0]
        loss_prev = recent[1].get("loss") if len(recent) > 1 else None
        runs.append({
            "name": run_dir.name,
            "step": last.get("step") or 0,
            "total": last.get("total") or 0,
            "epoch": last.get("epoch") or 0,
            "loss": last.get("loss"),
            "loss_prev": loss_prev,
            "lr": last.get("lr"),
            "rps": last.get("recs_per_s") or 0.0,
            "gpu_gb": last.get("gpu_gb") or 0.0,
            "mtime": os.path.getmtime(path),
            "active": any("clarke-train" in c and (str(run_dir) in c or run_dir.name in c)
                          for c in cmds),
        })
    runs.sort(key=lambda r: (not r["active"], -r["mtime"]))
    return runs


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
    """Eighth-resolution bar so slow movers still visibly advance."""
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
    s = "".join(SPARK[min(7, int(v / peak * 7))] for v in vals)
    return s.rjust(width)


def eta_str(run: dict, bs: int = 6, accum: int = 2) -> str:
    if not run["total"] or run["step"] >= run["total"] or run["rps"] <= 0:
        return "—"
    secs = (run["total"] - run["step"]) / (run["rps"] / (bs * accum))
    if secs >= 3600:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 60:.0f}m"


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
        self.log_idx = 0

    # -- primitives
    def put(self, win, y: int, x: int, text: str, attr: int = 0) -> None:
        """Clip by CHARACTERS, not bytes: box-drawing glyphs are 3 bytes in UTF-8 and
        addnstr() counts bytes, which silently cut every frame to a third of its width."""
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x > w - 1:
            return
        avail = w - x
        text = text[:avail]
        if not text:
            return
        try:
            win.addstr(y, x, text, attr)
        except curses.error:
            # writing the very last cell can raise; drop one glyph and retry
            try:
                win.addstr(y, x, text[:-1], attr)
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

    # -- frame
    def frame(self, win) -> None:
        win.erase()
        h, w = win.getmaxyx()
        now = time.time()

        runs = discover_runs()
        gpu = gpu_stats()
        cores, cpust = cpu_usage(self.cpu_prev)
        self.cpu_prev = cpust["prev"]
        load = cpust["load"]
        mem = mem_stats()
        counters = net_counters()
        dt = max(1e-3, now - self.net_t)
        rates = {n: (max(0.0, (rx - self.net_prev[n][0]) / dt),
                     max(0.0, (tx - self.net_prev[n][1]) / dt))
                 for n, (rx, tx) in counters.items() if self.net_prev and n in self.net_prev}
        self.net_prev, self.net_t = counters, now
        if cores:
            self.h_cpu.append(sum(cores) / len(cores))
        if gpu:
            self.h_gpu.append(gpu["util"])
        if rates:
            self.h_net.append(sum(r + t for r, t in rates.values()))

        active = next((r for r in runs if r["active"]), runs[0] if runs else None)
        y = 0

        # ── header ────────────────────────────────────────────────────────
        self.put(win, y, 1, "clarke-flash", curses.color_pair(C_ACCENT) | curses.A_BOLD)
        self.put(win, y, 14, time.strftime("%H:%M:%S"), curses.A_BOLD)
        status = "ACTIVE" if active and active["active"] else ("IDLE" if not active else "DONE")
        attr = curses.color_pair(C_GOOD if status == "ACTIVE" else C_WARN) | curses.A_BOLD
        self.put(win, y, w - len(status) - 3, f"● {status}", attr)
        y += 1

        # ── active run block ──────────────────────────────────────────────
        if active:
            pct = 100.0 * active["step"] / active["total"] if active["total"] else 0.0
            name = active["name"][:24]
            self.put(win, y, 1, f"run", curses.color_pair(C_DIM))
            self.put(win, y, 7, name, curses.A_BOLD)
            step_txt = f"step {active['step']:,} / {active['total']:,}"
            self.put(win, y, max(34, w // 2), step_txt, curses.color_pair(C_DIM))
            y += 1

            bw = max(20, w - 26)
            self.put(win, y, 7, pbar(pct, bw), curses.color_pair(colour_for(pct)))
            self.put(win, y, 8 + bw, f" {pct:5.1f}%", curses.color_pair(colour_for(pct)))
            y += 1

            loss = active["loss"]
            d = ""
            prev_loss = active.get("loss_prev")
            if loss is not None and isinstance(prev_loss, float):
                delta = loss - prev_loss
                d = f" {'↓' if delta < -0.05 else ('↑' if delta > 0.05 else '·')}"
            latt = curses.color_pair(C_GOOD if d.startswith(" ↓") else
                                     (C_WARN if d.startswith(" ↑") else C_ACCENT))
            self.put(win, y, 7, f"loss {loss:.3f}{d}", latt)
            self.put(win, y, 7 + len(f"loss {loss:.3f}{d}") + 1,
                     f"lr {active['lr']:.2e}   {active['rps']:5.1f} rec/s   "
                     f"ep {active['epoch']}   eta {eta_str(active)}", curses.color_pair(C_DIM))
            y += 1
        y += 0
        self.hline(win, y, 0, w)
        y += 1

        # ── geometry ─────────────────────────────────────────────────────
        half = (w - 3) // 2
        right_x, right_w = half + 3, w - half - 3
        cell = 4
        per_row = max(1, (right_w - 4) // cell)
        cores_shown = min(32, os.cpu_count() or 8)
        core_rows = max(1, (cores_shown + per_row - 1) // per_row)
        mid_h = max(6, min(3 + core_rows + 1, 9))

        # ── GPU ───────────────────────────────────────────────────────────
        self.box(win, y, 0, mid_h, half + 1, "GPU")
        gw = max(8, half - 34)
        if gpu:
            self.gauge(win, y + 1, 2, "util", gpu["util"], gw, "busy")
            vp = 100.0 * gpu["mem"] / max(gpu["mem_total"], 1)
            self.gauge(win, y + 2, 2, "vram", vp, gw,
                       f"{gpu['mem'] / 1024:.1f}/{gpu['mem_total'] / 1024:.0f} GiB")
            pw = (f"{gpu['power']:.0f}/{gpu['power_limit']:.0f} W"
                  if gpu.get("power_limit") else f"{gpu['power']:.0f} W")
            tattr = curses.color_pair(C_BAD if gpu["temp"] >= 85 else
                                      (C_WARN if gpu["temp"] >= 75 else C_GOOD))
            self.put(win, y + 3, 2, f"{'temp':<8}", curses.color_pair(C_DIM))
            self.put(win, y + 3, 10, f"{gpu['temp']:.0f}°C", tattr)
            self.put(win, y + 3, max(20, half - 22), pw, curses.color_pair(C_DIM))
            if mid_h >= 6:
                self.put(win, y + mid_h - 2, 2,
                         "util " + spark(list(self.h_gpu), max(8, half - 9)),
                         curses.color_pair(C_ACCENT))
        else:
            self.put(win, y + 1, 2, "nvidia-smi unavailable", curses.color_pair(C_BAD))

        # ── CPU ───────────────────────────────────────────────────────────
        self.box(win, y, right_x, mid_h, right_w, "CPU")
        if cores:
            total = sum(cores) / len(cores)
            cw = max(8, right_w - 32)
            self.gauge(win, y + 1, right_x + 2, "total", total, cw,
                       f"load {load[0]:.2f}")
            # per-core grid: fixed 4-char cells, wrapped to the panel width
            for i, u in enumerate(cores[:32]):
                ry = y + 2 + i // per_row
                if ry >= y + mid_h - 1:
                    break
                cx = right_x + 2 + (i % per_row) * cell
                self.put(win, ry, cx, pbar(u, 3), curses.color_pair(colour_for(u)))
            if mid_h >= 3 + core_rows + 1:
                self.put(win, y + mid_h - 2, right_x + 2,
                         "cpu " + spark(list(self.h_cpu), max(6, right_w - 8)),
                         curses.color_pair(C_ACCENT))
        else:
            self.put(win, y + 1, right_x + 2, "sampling…", curses.color_pair(C_DIM))
        y += mid_h

        # ── memory / network ──────────────────────────────────────────────
        low_h = 6
        self.box(win, y, 0, low_h, half + 1, "MEMORY")
        mw = max(8, half - 34)
        mp = 100.0 * mem["used"] / max(mem["total"], 1)
        self.gauge(win, y + 1, 2, "mem", mp, mw, f"{mem['used']:.1f}/{mem['total']:.1f} GB")
        sp = 100.0 * mem["swap_used"] / mem["swap_total"] if mem["swap_total"] else 0.0
        self.gauge(win, y + 2, 2, "swap", sp, mw,
                   f"{mem['swap_used']:.1f}/{mem['swap_total']:.1f} GB")
        disk = disk_stats()
        self.gauge(win, y + 3, 2, "disk", disk["pct"], mw,
                   f"{disk['free']:.0f} GiB free")

        self.box(win, y, right_x, low_h, right_w, "NETWORK")
        self.put(win, y + 1, right_x + 2, f"{'iface':<10}{'rx':>9}{'tx':>9}",
                 curses.color_pair(C_DIM) | curses.A_BOLD)
        top = sorted(rates.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:3]
        for i, (name, (rx, tx)) in enumerate(top[: low_h - 3]):
            self.put(win, y + 2 + i, right_x + 2,
                     f"{name[:9]:<10}{hb(rx) + '/s':>9}{hb(tx) + '/s':>9}",
                     curses.color_pair(C_GOOD))
        if not top:
            self.put(win, y + 2, right_x + 2, "sampling…", curses.color_pair(C_DIM))
        y += low_h

        # ── runs table ────────────────────────────────────────────────────
        remaining = h - y - 1
        log_h = 7 if remaining >= 11 else 0        # only steal space when both fit
        max_rows = 8
        table_h = max(3, min(min(len(runs), max_rows) + 3, remaining - log_h))
        if self.box(win, y, 0, table_h, w, "RUNS"):
            more = len(runs) - max(0, table_h - 3)
            cols = f"{'run':<18}{'step':>13}{'ep':>4}{'loss':>7}{'lr':>10}{'rec/s':>8}" \
                   f"{'vram':>7}   {'state':<11}"
            self.put(win, y + 1, 2, cols, curses.color_pair(C_DIM) | curses.A_BOLD)
            for i, r in enumerate(runs[: table_h - 4]):
                ago = int(now - r["mtime"])
                state = "● active" if r["active"] else f"{ago}s ago"
                attr = (curses.color_pair(C_GOOD) | curses.A_BOLD if r["active"] else 0)
                lr = f"{r['lr']:.1e}" if r["lr"] else "—"
                progress = f"{r['step']:,}/{r['total']:,}"
                self.put(win, y + 2 + i, 2,
                         f"{r['name'][:17]:<18}{progress:>13}"
                         f"{r['epoch']:>4}{(r['loss'] or 0):>7.3f}{lr:>10}"
                         f"{r['rps']:>8.1f}{r['gpu_gb']:>6.1f}G   {state:<11}", attr)
            hidden = len(runs) - min(len(runs), table_h - 4)
            if hidden > 0:
                self.put(win, y + table_h - 2, 2,
                         f"… {hidden} older run{'s' if hidden > 1 else ''} hidden",
                         curses.color_pair(C_DIM))
            y += table_h

        # ── log tail ──────────────────────────────────────────────────────
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
                    attr = 0
                    if "failed" in line.lower() or "error" in line.lower():
                        attr = curses.color_pair(C_BAD)
                    elif "done" in line.lower() or "trained" in line.lower():
                        attr = curses.color_pair(C_GOOD)
                    self.put(win, y + 1 + i, 2, line, attr or curses.color_pair(C_DIM))

        # ── status bar ────────────────────────────────────────────────────
        pause_txt = " · PAUSED" if self.paused else ""
        bar = (" q quit   space pause   l log   r reset   ? help "
               f"  [{self.interval:g}s{pause_txt}] ")
        self.put(win, h - 1, 0, bar.ljust(w - 1), curses.color_pair(C_ACCENT) | curses.A_REVERSE)
        win.noutrefresh()
        curses.doupdate()

    def help(self, win) -> None:
        lines = ["keys", "", "q / esc   quit", "space     pause refresh",
                 "l         next log file", "r         reset samplers", "?         this help",
                 "", "panels: GPU, CPU (per core), MEMORY, NETWORK, RUNS, LOG",
                 "runs are auto-discovered from runs/*/train.jsonl"]
        self.box(win, 2, 3, len(lines) + 4, 58, "HELP")
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
        _, first = cpu_usage(None)          # seed the delta samplers
        self.cpu_prev = first["prev"]
        self.net_prev = net_counters()
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
    runs = discover_runs()
    gpu = gpu_stats()
    mem = mem_stats()
    _, first = cpu_usage(None)                 # need a delta before any usage is meaningful
    time.sleep(0.2)
    cores, _ = cpu_usage(first["prev"])
    lines = [f"clarke-flash {time.strftime('%H:%M:%S')}"]
    for r in runs[:6]:
        tag = "● active" if r["active"] else f"{int(time.time() - r['mtime'])}s ago"
        lines.append(f"{r['name']:<16}{r['step']:>6}/{r['total']:<6} ep{r['epoch']} "
                     f"loss {r['loss']:.3f}  {r['rps']:5.1f} r/s  {r['gpu_gb']:4.1f}G  {tag}")
    if gpu:
        lines.append(f"gpu  util {gpu['util']:.0f}%  "
                     f"vram {gpu['mem'] / 1024:.1f}/{gpu['mem_total'] / 1024:.0f}G"
                     f"  {gpu['temp']:.0f}C  {gpu['power']:.0f}W")
    else:
        lines.append("gpu  nvidia-smi unavailable")
    cpu_pct = f"{sum(cores) / len(cores):.0f}%" if cores else "—"
    lines.append(f"mem  {mem['used']:.1f}/{mem['total']:.1f}G  "
                 f"swap {mem['swap_used']:.1f}/{mem['swap_total']:.1f}G  "
                 f"cpu {cpu_pct}  cores {os.cpu_count()}")
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
