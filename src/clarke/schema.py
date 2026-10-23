"""Typed field schema for the Clarke extraction model.

Defines the field types Clarke answers (mirroring Cloudflare's extraction
model API), how every type's answer space is decomposed into categorical
grids the head can predict (spans, sign/exponent/digits, calendar components,
binned distributions), and how those predictions are rendered back into
typed answers with confidence, ``p_null`` and percentiles.

Answer-space sizes are fixed per head:

text/list spans        pointer logits over state tokens (+1 null slot)
number/duration        sign(2) x exponent(13) x top3(1000) x next3(1000)
date                 year(256) x month(12) x day(31)
time                 hour(24) x minute(60) x second(60)
estimate/forecast    grid bins (<= 256) + below + above slots
"""

from __future__ import annotations

import calendar
import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any

FIELD_TYPES = ("text", "number", "date", "time", "datetime", "duration", "list", "estimate", "forecast")

TYPE_IDS = {name: index for index, name in enumerate(
    ["text", "number", "date", "time", "datetime", "duration",
     "list-text", "list-number", "list-date", "estimate", "forecast"]
)}

# number head bins
SIGN_BINS = 2
EXP_BINS = 13          # decimal exponent of the scaled integer, 0..12
DIGIT3_BINS = 1000     # top three digits, then next three digits

# date head bins
YEAR_BINS = 256
YEAR_MIN = 1900
MONTH_BINS = 12
DAY_BINS = 31          # 1..31, clamped to month length on decode
HOUR_BINS = 24
MIN_BINS = 60
SEC_BINS = 60

# distribution head bins (estimate/forecast)
MAX_GRID_BINS = 256
OVERFLOW_BINS = 2      # p_below, p_above appended after grid bins

LIST_SLOTS = 8         # max items for list fields
SPAN_MAX_TOKENS = 256   # max span length in tokens for text/list spans


class FieldSpec:
    """A named typed field with its answer grid."""

    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self.type = str(spec["type"])
        if self.type not in FIELD_TYPES:
            raise ValueError(f"{name}: unknown field type {self.type!r}")
        self.instructions = spec.get("instructions") or name
        self.unit = spec.get("unit")
        self.integer = bool(spec.get("integer", False))
        self.decimals = 0 if self.integer else int(spec.get("decimals", 2))
        self.items = str(spec.get("items", "text")) if self.type == "list" else None
        self.min = float(spec["min"]) if spec.get("min") is not None else None
        self.max = float(spec["max"]) if spec.get("max") is not None else None
        self.step = float(spec["step"]) if spec.get("step") is not None else None
        self.scale = str(spec.get("scale", "linear")) if self.type in ("estimate", "forecast") else "linear"
        self.horizon = int(spec.get("horizon", 1)) if self.type == "forecast" else 0
        if self.type in ("estimate", "forecast"):
            if self.min is None or self.max is None or self.step is None:
                raise ValueError(f"{name}: estimate/forecast require min, max and step")
            if not self.max > self.min:
                raise ValueError(f"{name}: require max > min")

    @property
    def type_id(self) -> int:
        if self.type == "list":
            return TYPE_IDS[f"list-{self.items}"]
        return TYPE_IDS[self.type]

    def config(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (
            ("unit", self.unit), ("integer", self.integer or None),
            ("decimals", self.decimals if self.type == "number" else None),
            ("items", self.items), ("min", self.min), ("max", self.max),
            ("step", self.step), ("scale", self.scale if self.type in ("estimate", "forecast") else None),
            ("horizon", self.horizon or None),
        ):
            if value is not None:
                out[key] = value
        return out

    def render(self, index: int) -> str:
        config = self.config()
        block = f"\nFIELD {index}\nID: {self.name}\nTYPE: {self.type}\n"
        if config:
            block += f"CONFIG: {json.dumps(config, separators=(',', ':'))}\n"
        block += f"INSTRUCTION: {self.instructions}\nEND FIELD\n"
        return block

    # -- grids -------------------------------------------------------------

    def grid_centers(self) -> list[float]:
        """Center values of the categorical answer grid (estimate/forecast).

        If the user's grid fits in MAX_GRID_BINS steps we bin it exactly;
        otherwise a log-spaced (or linear) grid inside [min, max]. The two
        trailing slots are below/above overflow.
        """
        if self.type not in ("estimate", "forecast"):
            raise ValueError(self.type)
        span = self.max - self.min
        steps = int(round(span / self.step)) + 1
        if steps <= MAX_GRID_BINS:
            edges = [self.min + i * self.step for i in range(steps + 1)]
        else:
            edges = self.bounds()
        centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
        return centers + [self.min - span, self.max + span]

    def bounds(self, bins: int = MAX_GRID_BINS) -> list[float]:
        if self.scale == "log" and self.min > 0:
            lo, hi = math.log(self.min), math.log(self.max)
            return [math.exp(lo + (hi - lo) * i / bins) for i in range(bins + 1)]
        return [self.min + (self.max - self.min) * i / bins for i in range(bins + 1)]

    def bin_value(self, value: float) -> int:
        """Target bin for a realized value on this field's grid."""
        centers = self.grid_centers()
        grid = centers[: -OVERFLOW_BINS]
        if value < self.min:
            return len(centers) - OVERFLOW_BINS
        if value > self.max:
            return len(centers) - 1
        steps = int(round((self.max - self.min) / self.step)) + 1
        if steps == len(grid):
            index = int(round((value - self.min) / self.step))
        elif self.scale == "log" and self.min > 0:
            lo, hi = math.log(self.min), math.log(self.max)
            index = int((math.log(max(value, 1e-12)) - lo) / (hi - lo) * len(grid))
        else:
            index = int((value - self.min) / (self.max - self.min) * len(grid))
        return min(max(index, 0), len(grid) - 1)

    def quantiles(self, probabilities: list[float]) -> dict[str, Any]:
        """Median and p10/p90 (80% interval) from a distribution over grid centers."""
        centers = self.grid_centers()
        p_below = probabilities[len(centers) - OVERFLOW_BINS]
        p_above = probabilities[len(centers) - OVERFLOW_BINS + 1]
        grid = centers[: -OVERFLOW_BINS]
        weights = probabilities[: len(grid)]
        total = sum(weights) or 1.0
        cdf: list[float] = []
        running = 0.0
        for weight in weights:
            running += weight / total
            cdf.append(running)

        def quantile(q: float) -> float:
            if q < p_below:
                return grid[0]
            if q > 1.0 - p_above:
                return grid[-1]
            for index, cdf_value in enumerate(cdf):
                if cdf_value >= q:
                    return grid[index]
            return grid[-1]

        return {
            "median": quantile(0.5),
            "p10": quantile(0.1),
            "p90": quantile(0.9),
            "p_below": round(p_below, 4),
            "p_above": round(p_above, 4),
        }


# -- number / duration decomposition ----------------------------------------


def decompose_number(value: float, decimals: int) -> tuple[int, int, int, int]:
    """(sign, exponent, top3, next3) target bins for the integer |v|*10^decimals.

    ``exponent`` is the decimal exponent of that integer (digits - 1). ``top3``
    holds the leading three digits (zero-padded to three digits when fewer),
    ``next3`` the following three (zero-padded when fewer). Decoded value is
    ``top3 * 10^(e-2) + next3 * 10^(e-5)`` — exact up to six significant
    digits, then ~0.1% relative resolution.
    """
    scaled = int(round(abs(value) * (10 ** decimals)))
    sign = 0 if value < 0 else 1
    if scaled == 0:
        return sign, 0, 0, 0
    digits = len(str(scaled))
    exponent = min(digits - 1, EXP_BINS - 1)
    if digits <= 3:
        top, next3 = scaled * 10 ** (3 - digits), 0
    else:
        top = scaled // 10 ** (digits - 3)
        remainder = scaled % 10 ** (digits - 3)
        next3 = remainder // 10 ** (digits - 6) if digits >= 6 else remainder * 10 ** (6 - digits)
    return sign, exponent, min(top, 999), min(next3, 999)


def render_number(sign_bin: int, exp_bin: int, top_bin: int, next_bin: int, decimals: int) -> float:
    if exp_bin == 0 and top_bin == 0 and next_bin == 0:
        return 0.0
    top = top_bin * (10 ** max(exp_bin + 1 - 3, 0)) if exp_bin >= 2 else top_bin // (10 ** (2 - exp_bin))
    nxt = next_bin * (10 ** max(exp_bin + 1 - 6, 0)) if exp_bin >= 5 else next_bin // (10 ** (5 - exp_bin))
    scaled = max(0, top + nxt)
    value = scaled / (10 ** decimals)
    return -value if sign_bin == 0 else value


def clamp_date(year_bin: int, month: int, day: int) -> dt.date:
    year = YEAR_MIN + year_bin
    year = max(1, min(year, 9999))
    month = min(max(month, 1), 12)
    day = min(max(day, 1), calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)
