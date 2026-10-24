"""Synthetic record generator with exact ground truth by construction.

Every document is assembled from fragments, so the character span of each
answer is known while the text is written — same for computed values (totals,
date arithmetic), comparable-sales estimates and forecast series (labels are
realized draws from the generator's latent process).

Domains: support email, invoice, ticket, table (aggregates), comparables
(estimate), time series (forecast). Distractors and absent fields teach
``p_null``; format variation teaches grounded reading over pattern matching.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
import re
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from .candidates import parse_date

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FIRST_NAMES = ["Dana", "Priya", "Marcus", "Yusuf", "Elena", "Jonah", "Amara", "Tomas", "Ingrid",
               "Leo", "Maya", "Owen", "Sofia", "Raj", "Nina", "Felix", "Zara", "Hugo"]
LAST_NAMES = ["Whitfield", "Okafor", "Lindqvist", "Moreau", "Kaplan", "Nguyen", "Alvarez",
              "Berg", "Sato", "Kowalski", "Ferreira", "Haddad", "Novak", "Silva"]
COMPANIES = ["Acme Supply", "Northwind Traders", "Bluepeak Logistics", "Orchard Labs",
             "Harbor Freight Co", "Meridian Works", "Cedar & Stone", "Quantum Prints",
             "Fairfax Plumbing", "Lumen Electrical", "Redwood Movers", "Atlas Couriers"]
VENDORS = ["Staples", "SysTools Inc", "Grainger", "OfficeMax", "Dell Technologies",
           "Uline", "McMaster-Carr", "ADP Supplies", "Fancy Foods", "TechDepot"]
PRODUCTS = ["27-inch monitor", "laptop stand", "ergonomic chair", "USB-C hub", "webcam",
            "desk lamp", "keyboards", "headset", "surge protector", "standing desk",
            "office chair", "monitor arm", "whiteboard", "router"]
DAMAGES = ["cracked screen", "dented corner", "scratched housing", "missing power cable",
           "dead pixels", "bent legs", "frayed charging port", "wobbly base",
           "faulty switch", "leaking battery"]
STATUSES = ["delivered", "damaged on arrival", "still in transit", "refused", "returned"]
ISSUES = ["charged twice", "order never arrived", "wrong item shipped", "late delivery",
          "defective unit", "missing parts"]

ORDERS = ["A-88412", "K-10293", "Z4471", "77245", "9-556-AB", "1Z-998-221"]


def format_date(rng: random.Random, d: date) -> str:
    style = rng.choice(["iso", "us", "month", "long", "slash"])
    if style == "iso":
        return d.isoformat()
    if style == "us":
        return f"{d.month}/{d.day}/{d.year}"
    if style == "month":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == "long":
        return d.strftime("%d %B %Y").lstrip("0")
    return f"{d.month:02d}/{d.day:02d}/{d.year % 100:02d}"


def format_money(rng: random.Random, value: float) -> str:
    style = rng.choice(["plain", "comma", "bare"])
    if style == "plain":
        return f"${value:.2f}"
    if style == "comma":
        return f"${value:,.2f}"
    return f"{value:.2f}"


class Builder:
    """Assembles state text while recording character spans for answers."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: dict[str, tuple[int, int]] = {}
        self.item_spans: dict[str, list[tuple[int, int]]] = {}

    def plain(self, text: str) -> None:
        self.parts.append(text)

    def value(self, field: str, text: str, item: bool = False, index: int = 0) -> None:
        start = sum(len(part) for part in self.parts)
        self.parts.append(text)
        if item:
            self.item_spans.setdefault(field, [])
            while len(self.item_spans[field]) <= index:
                self.item_spans[field].append((0, 0))
            self.item_spans[field][index] = (start, start + len(text))
        else:
            self.spans[field] = (start, start + len(text))

    def text(self) -> str:
        return "".join(self.parts)


def make_record(builder: Builder, fields: dict[str, Any], answers: dict[str, Any],
                domain: str, index: int, rng: random.Random) -> dict[str, Any]:
    """Randomize surface names/instructions, then attach grounded spans.

    Field IDs and instructions are the model's only handle on what to answer,
    so both get shuffled per record (meaning-preserving): the head must read
    the schema, not pattern-match fixed keys.
    """
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for key in fields:
        new = surface_name(rng, key)
        if new in used:
            new = key
        used.add(new)
        mapping[key] = new
    fields_out: dict[str, Any] = {}
    answers_out: dict[str, Any] = {}
    spans: dict[str, tuple[int, int]] = {}
    item_spans: dict[str, list[tuple[int, int]]] = {}
    for key, spec in fields.items():
        new = mapping[key]
        spec = dict(spec)
        spec["instructions"] = paraphrase(rng, spec["instructions"])
        fields_out[new] = spec
        answers_out[new] = answers.get(key, {"value": None})
        if key in builder.spans:
            spans[new] = builder.spans[key]
        if key in builder.item_spans:
            item_spans[new] = builder.item_spans[key]
    builder.spans, builder.item_spans = spans, item_spans
    # Cloudflare permutes schema field order in their synthetic post-training;
    # order must never leak which field is which.
    order = list(fields_out)
    rng.shuffle(order)
    fields_out = {k: fields_out[k] for k in order}

    for name, spec in fields_out.items():
        answer = answers_out.setdefault(name, {"value": None})
        if spec["type"] == "text" and answer.get("value") is not None:
            if name in builder.spans:
                answer["span"] = list(builder.spans[name])
            else:
                raise AssertionError(f"{domain}-{index}: text field {name!r} has no registered span")
        if spec["type"] == "list" and spec.get("items") == "text" and answer.get("value"):
            answer["spans"] = [list(span) for span in item_spans.get(name, [])]
    state = builder.text()
    digest = hashlib.md5(state.encode()).hexdigest()[:6]
    return {
        "id": f"{domain}-{index}-{digest}",
        "domain": domain,
        "state": state,
        "fields": fields_out,
        "answers": answers_out,
    }


# -- domains ---------------------------------------------------------------


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.title() for word in rest)


def surface_name(rng: random.Random, key: str) -> str:
    """Randomize the field's surface ID — same meaning, many spellings.

    Teaches the head that the instruction/ID is something to read, not a
    lookup key: 70% of fields appear under a shuffled name form.
    """
    if rng.random() < 0.3:
        return key
    return rng.choice([
        key, _camel(key), key.replace("_", "-"), "the_" + key,
        key + "_value", key.upper(), key.replace("_", ""),
    ])


def paraphrase(rng: random.Random, text: str) -> str:
    """Same ask, different wording (and a little noise)."""
    return rng.choice([
        text,
        text.rstrip("?."),
        text.lower(),
        f"{text} (copy it exactly)",
        f"Find: {text.rstrip('?.')}?",
        f"Report {text[0].lower() + text[1:].rstrip('?.')}; it may be absent.",
    ])


def support_email(rng: random.Random, index: int) -> dict[str, Any]:
    builder = Builder()
    name = rng.choice(FIRST_NAMES)
    order = rng.choice(ORDERS)
    delivered = date(2026, rng.randint(7, 10), rng.randint(1, 27))
    wrote = delivered + timedelta(days=rng.randint(3, 45))
    price = round(rng.uniform(49, 1499), 2)
    tax = round(price * rng.uniform(0.04, 0.11), 2)
    total = round(price + tax, 2)
    damages = rng.sample(DAMAGES, rng.randint(1, 3))
    issue = rng.choice(ISSUES)

    builder.plain("Hi, order ")
    builder.value("order_id", order)
    builder.plain(" arrived on ")
    builder.value("delivered_on", format_date(rng, delivered))
    builder.plain(" with ")
    for position, damage in enumerate(damages):
        builder.value("damage", damage, item=True, index=position)
        builder.plain(" and " if position == 0 and len(damages) > 1 else
                      ", " if position < len(damages) - 1 else ". ")
    problem = rng.choice([
        "The hinge snapped within a week of delivery.",
        "It stopped charging overnight and will not power on now.",
        "The right side makes a grinding noise whenever it runs.",
        "Half the pixels went black the first afternoon.",
    ])
    builder.value("problem_sentence", problem)
    builder.plain(" I paid " + f"{format_money(rng, price)} plus {format_money(rng, tax)} tax. ")
    builder.plain(f"It's now {format_date(rng, wrote)} and I'd like this fixed or refunded. ")
    builder.plain("My order had the wrong billing address too — charged twice with the ")
    builder.plain(f"order reference {order} again. ")
    builder.plain("Thanks, ")
    builder.value("customer_name", name)

    tracking_present = rng.random() < 0.4
    if tracking_present:
        tracking = f"1Z{rng.randint(100, 999)}{rng.randint(10000, 99999)}"
        builder.plain(" The carrier gave me tracking ")
        builder.value("tracking_number", tracking)
        builder.plain(".")

    fields = {
        "order_id": {"type": "text", "instructions": rng.choice(
            ["Order number", "The order reference", "Order ID mentioned"])},
        "customer_name": {"type": "text", "instructions": "Customer's first name"},
        "delivered_on": {"type": "date", "instructions": "Delivery date"},
        "wrote_on": {"type": "date", "instructions": "Date the customer wrote"},
        "amount_paid": {"type": "number", "decimals": 2, "unit": "USD",
                        "instructions": "Total paid including tax"},
        "days_since_delivery": {"type": "duration", "unit": "days", "integer": True,
                                "instructions": "Days from delivery to writing"},
        "damage": {"type": "list", "items": "text", "instructions": "Each kind of damage described"},
    }
    answers = {
        "order_id": {"value": order},
        "customer_name": {"value": name},
        "delivered_on": {"value": delivered.isoformat()},
        "wrote_on": {"value": wrote.isoformat()},
        "amount_paid": {"value": total},
        "days_since_delivery": {"value": float((wrote - delivered).days)},
        "damage": {"value": damages},
        "customer_phone": {"value": None},
    }
    fields["customer_phone"] = {"type": "text", "instructions": "Customer phone number"}
    if rng.random() < 0.5:
        fields["tracking_number"] = {"type": "text", "instructions": "Carrier tracking number"}
        answers["tracking_number"] = {"value": tracking if tracking_present else None}
    fields["problem_sentence"] = {"type": "text",
                                  "instructions": "The sentence describing what went wrong"}
    answers["problem_sentence"] = {"value": problem}
    return make_record(builder, fields, answers, "support", index, rng)


def invoice(rng: random.Random, index: int) -> dict[str, Any]:
    builder = Builder()
    vendor = rng.choice(VENDORS)
    po = f"PO-{rng.randint(10000, 99999)}"
    invoice_date = date(2026, rng.randint(1, 10), rng.randint(1, 28))
    due = invoice_date + timedelta(days=rng.choice([14, 30, 45]))
    lines = [(rng.choice(PRODUCTS), rng.randint(1, 9), round(rng.uniform(5, 400), 2))
             for _ in range(rng.randint(2, 5))]
    subtotal = round(sum(q * p for _, q, p in lines), 2)
    tax_rate = rng.choice([0.05, 0.0725, 0.08, 0.1])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    builder.value("vendor", vendor)
    builder.plain(" — Invoice ")
    builder.value("invoice_id", po)
    builder.plain("\n")
    builder.plain(f"Date: ")
    builder.value("invoice_date", format_date(rng, invoice_date))
    builder.plain("\nPayment terms: net, due ")
    builder.value("due_date", format_date(rng, due))
    builder.plain("\n\nLine items:\n")
    for position, (product, qty, price) in enumerate(lines):
        builder.plain(f"- ")
        builder.value("items", product, item=True, index=position)
        builder.plain(f" x ")
        builder.value("quantities", str(qty), item=True, index=position)
        builder.plain(f" @ {format_money(rng, price)}\n")
    builder.plain(f"Subtotal: {format_money(rng, subtotal)}\n")
    builder.plain(f"Tax: ")
    builder.value("tax", format_money(rng, tax))
    builder.plain(f"\nTotal due: {format_money(rng, total)}\n")
    builder.plain(f"Remit to {vendor}, {rng.randint(100, 999)} Industrial Way.")

    fields = {
        "vendor": {"type": "text", "instructions": "Vendor name"},
        "invoice_id": {"type": "text", "instructions": "Invoice number"},
        "invoice_date": {"type": "date", "instructions": "Date of the invoice"},
        "due_date": {"type": "date", "instructions": "Payment due date"},
        "days_to_pay": {"type": "duration", "unit": "days", "integer": True,
                        "instructions": "Days between invoice date and due date"},
        "line_count": {"type": "number", "integer": True, "instructions": "Number of line items"},
        "subtotal": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Subtotal"},
        "tax": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Tax charged"},
        "tax_rate": {"type": "number", "decimals": 4, "instructions": "Tax rate as a decimal fraction"},
        "total_due": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Total amount due"},
        "items": {"type": "list", "items": "text", "instructions": "Each product ordered"},
        "quantities": {"type": "list", "items": "number", "integer": True,
                       "instructions": "Quantity of each product, in order"},
    }
    answers = {
        "vendor": {"value": vendor},
        "invoice_id": {"value": po},
        "invoice_date": {"value": invoice_date.isoformat()},
        "due_date": {"value": due.isoformat()},
        "days_to_pay": {"value": float((due - invoice_date).days)},
        "line_count": {"value": float(len(lines))},
        "subtotal": {"value": subtotal},
        "tax": {"value": tax},
        "tax_rate": {"value": tax_rate},
        "total_due": {"value": total},
        "items": {"value": [product for product, _, _ in lines]},
        "quantities": {"value": [float(qty) for _, qty, _ in lines]},
    }
    if rng.random() < 0.5:
        fields["purchase_order"] = {"type": "text", "instructions": "Customer purchase order number"}
        answers["purchase_order"] = {"value": None}
    # cross-referential fields: answers only obtainable by following the instruction
    fields["later_date"] = {"type": "date",
                            "instructions": "The later of the two dates in this document"}
    answers["later_date"] = {"value": due.isoformat()}
    fields["first_line_product"] = {"type": "text", "instructions": "Product on the first line item"}
    answers["first_line_product"] = {"value": lines[0][0]}
    builder.spans["first_line_product"] = builder.item_spans["items"][0]
    return make_record(builder, fields, answers, "invoice", index, rng)


def ticket(rng: random.Random, index: int) -> dict[str, Any]:
    builder = Builder()
    reporter = rng.choice(FIRST_NAMES) + " " + rng.choice(LAST_NAMES)
    created = datetime(2026, rng.randint(6, 10), rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59))
    resolved = created + timedelta(minutes=rng.randint(30, 5000))
    severity = rng.choice(["low", "medium", "high", "critical"])
    team = rng.choice(["billing", "technical", "logistics", "warehouse"])
    cost = round(rng.uniform(0, 850), 2)
    product = rng.choice(PRODUCTS)

    builder.plain(f"Ticket #{rng.randint(10000, 99999)}\n")
    builder.plain("Opened by ")
    builder.value("reporter", reporter)
    builder.plain(" at ")
    builder.value("opened_at", created.strftime("%Y-%m-%dT%H:%M:00"))
    builder.plain("\nSeverity: ")
    builder.value("severity", severity)
    builder.plain("\nAssigned team: ")
    builder.value("team", team)
    builder.plain("\nStatus update: resolved ")
    builder.value("resolved_at", resolved.strftime("%Y-%m-%dT%H:%M:00"))
    builder.plain("\nRepair cost on the damaged ")
    builder.value("product", product)
    builder.plain(f" came to {format_money(rng, cost)}.")

    fields = {
        "reporter": {"type": "text", "instructions": "Full name of the person who opened the ticket"},
        "opened_at": {"type": "datetime", "instructions": "When the ticket was opened"},
        "resolved_at": {"type": "datetime", "instructions": "When the ticket was resolved"},
        "resolution_time": {"type": "duration", "unit": "hours", "decimals": 1,
                            "instructions": "Hours from opening to resolution"},
        "severity": {"type": "text", "instructions": "Severity level"},
        "team": {"type": "text", "instructions": "Team assigned"},
        "repair_cost": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Repair cost"},
        "product": {"type": "text", "instructions": "Product involved"},
        "customer_phone": {"type": "text", "instructions": "Customer phone number"},
    }
    answers = {
        "reporter": {"value": reporter},
        "opened_at": {"value": created.strftime("%Y-%m-%dT%H:%M:00")},
        "resolved_at": {"value": resolved.strftime("%Y-%m-%dT%H:%M:00")},
        "resolution_time": {"value": round((resolved - created).total_seconds() / 3600, 1)},
        "severity": {"value": severity},
        "team": {"value": team},
        "repair_cost": {"value": cost},
        "product": {"value": product},
        "customer_phone": {"value": None},
    }
    return make_record(builder, fields, answers, "ticket", index, rng)


def table_record(rng: random.Random, index: int) -> dict[str, Any]:
    builder = Builder()
    rows = []
    region_spans: list[tuple[int, int]] = []
    base = date(2026, rng.randint(1, 6), 1)
    builder.plain("sales export, csv\n")
    builder.plain("date,region,units,revenue\n")
    for i in range(rng.randint(5, 14)):
        day = base + timedelta(days=i)
        units = rng.randint(10, 400)
        revenue = round(units * rng.uniform(2, 18), 2)
        region = rng.choice(["north", "south", "east", "west"])
        rows.append((day, region, units, revenue))
        builder.plain(f"{day.isoformat()},")
        start = len(builder.text())
        builder.plain(region)
        region_spans.append((start, len(builder.text())))
        builder.plain(f",{units},{revenue:.2f}\n")
    builder.plain("end of export")

    units = [float(r[2]) for r in rows]
    revenues = [r[3] for r in rows]
    max_i = max(range(len(rows)), key=lambda i: rows[i][3])
    builder.spans["max_revenue_region"] = region_spans[max_i]
    fields = {
        "row_count": {"type": "number", "integer": True, "instructions": "Number of data rows"},
        "total_units": {"type": "number", "integer": True, "instructions": "Sum of the units column"},
        "average_revenue": {"type": "number", "decimals": 2, "instructions": "Mean of the revenue column"},
        "max_revenue_region": {"type": "text", "instructions": "Region of the row with highest revenue"},
        "total_revenue": {"type": "number", "decimals": 2, "unit": "USD", "instructions": "Sum of revenue"},
    }
    answers = {
        "row_count": {"value": float(len(rows))},
        "total_units": {"value": sum(units)},
        "average_revenue": {"value": round(sum(revenues) / len(revenues), 2)},
        "max_revenue_region": {"value": rows[max_i][1]},
        "total_revenue": {"value": round(sum(revenues), 2)},
    }
    return make_record(builder, fields, answers, "table", index, rng)


def comparables(rng: random.Random, index: int) -> dict[str, Any]:
    """Home-prices style regression state; estimate label is a realized draw."""
    builder = Builder()
    intercept = rng.uniform(120000, 320000)
    per_sqft = rng.uniform(120, 340)
    per_bed = rng.uniform(5000, 30000)
    age_penalty = rng.uniform(500, 2400)

    def price(sqft: float, beds: float, year: float, neighborhood: str) -> float:
        premium = {"Elm Park": 1.1, "Riverside": 0.85, "Hillcrest": 1.25}.get(neighborhood, 1.0)
        value = (intercept + per_sqft * sqft + per_bed * beds - age_penalty * (2026 - year)) * premium
        return max(20000, value + rng.gauss(0, value * 0.06))

    rows = []
    neighborhoods = list({"Elm Park", "Riverside", "Hillcrest"})
    for _ in range(rng.randint(4, 9)):
        sqft = rng.randint(700, 3200)
        beds = rng.randint(1, 5)
        year = rng.randint(1960, 2024)
        hood = rng.choice(neighborhoods)
        sale = round(price(sqft, beds, year, hood), -3)
        rows.append((sqft, beds, year, hood, sale))
    subject = (rng.randint(700, 3200), rng.randint(1, 5), rng.randint(1960, 2024), rng.choice(neighborhoods))
    true_price = round(price(*subject), -3)

    builder.plain("Comparable sales:\nsqft,beds,year_built,neighborhood,sale_price\n")
    for sqft, beds, year, hood, sale in rows:
        builder.plain(f"{sqft},{beds},{year},{hood},{int(sale)}\n")
    builder.plain("\nSubject property:\nsqft,beds,year_built,neighborhood\n")
    builder.plain(f"{subject[0]},{subject[1]},{subject[2]},{subject[3]}\n")

    fields = {
        "sale_price": {"type": "estimate", "unit": "USD", "min": 50000, "max": 2500000,
                       "scale": "log", "step": 1000,
                       "instructions": "Expected sale price of the subject property"},
    }
    answers = {"sale_price": {"value": true_price}}
    return make_record(builder, fields, answers, "comps", index, rng)


def forecast_record(rng: random.Random, index: int) -> dict[str, Any]:
    builder = Builder()
    horizon = rng.randint(1, 4)
    days = rng.randint(28, 90)
    start = date(2026, rng.randint(1, 8), rng.randint(1, 28))
    level = rng.uniform(5e5, 4e6) if rng.random() < 0.5 else rng.uniform(50, 5000)
    trend = rng.uniform(-0.002, 0.002)
    weekly = [rng.uniform(0.85, 0.95), rng.uniform(0.8, 0.95), rng.uniform(0.9, 1.0),
              rng.uniform(1.0, 1.15), rng.uniform(1.0, 1.2), rng.uniform(0.75, 0.9),
              rng.uniform(0.9, 1.1)]
    scale = rng.uniform(0.01, 0.06)

    def draw(day: date, step: int) -> float:
        base = level * (1 + trend * step) * weekly[day.weekday()]
        return max(1.0, base * (1 + rng.gauss(0, scale)))

    history = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        history.append((day, draw(day, offset)))
    future = [(start + timedelta(days=days + h), draw(start + timedelta(days=days + h), days + h))
              for h in range(horizon)]

    kind, setting = rng.choice([("travelers", "US airport checkpoints"), ("calls", "the contact center"),
                                ("shipments", "the warehouse")])
    builder.plain(f"Daily number of {kind} at {setting}.\n")
    builder.plain("date,weekday,value\n")
    for day, value in history:
        builder.plain(f"{day.isoformat()},{['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day.weekday()]},{int(round(value))}\n")
    last = history[-1][0]
    next_day = last + timedelta(days=1)
    builder.plain(f"Forecast dates: {(next_day).isoformat()} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][next_day.weekday()]})")
    if horizon > 1:
        builder.plain(f" through {(start + timedelta(days=days + horizon - 1)).isoformat()}")
    builder.plain(".")

    lo, hi = min(v for _, v in history), max(v for _, v in history)
    margin = (hi - lo) * 0.35 + hi * 0.1
    fields = {
        "value_next": {"type": "forecast", "horizon": horizon, "interval": "day",
                       "min": max(0, lo - margin), "max": hi + margin,
                       "scale": "log" if lo > 0 else "linear",
                       "step": max(1.0, round((hi + margin - max(0, lo - margin)) / 200)),
                       "instructions": f"Number of {kind} for each of the next {horizon} days"},
    }
    answers = {"value_next": {"values": [round(v, 2) for _, v in future]}}
    return make_record(builder, fields, answers, "forecast", index, rng)


# --- contract domain (CUAD-shaped) ----------------------------------------
# Field names match section headers; answers sit below (sometimes several
# paragraphs below) the matching header; NEAR-MISS decoy sections carry
# similar titles but different content; ~40% of requested fields are absent;
# documents are 5-12k chars with 20-40 schema fields. This is the data shape
# every existing domain lacks: the binding task is header SEMANTICS, not
# adjacency (journal 07:00 diagnosis).

_P = "paragraph"; _M = "money"; _D = "date"; _U = "duration"; _L = "list-addr"

CLAUSE_KINDS: dict[str, str] = {
    "confidentiality_obligations": _P, "assignment_restriction": _P,
    "audit_rights": _P, "warranty_disclaimer": _P, "indemnification_scope": _P,
    "non_solicitation_terms": _P, "force_majeure_carveout": _P, "ip_ownership": _P,
    "late_fee_amount": _M, "liquidated_damages_cap": _M,
    "insurance_minimum_coverage": _M, "early_termination_fee": _M,
    "renewal_date": _D, "notice_deadline": _D,
    "cure_period": _U, "data_retention_period": _U,
    "notice_addresses": _L,
}

DECOY_TITLES = ["Termination for Cause", "Confidentiality Exclusions",
                "Assignment and Subcontracting", "Audit Committee Procedures",
                "Limited Product Warranty", "Indemnification Procedures",
                "Solicitation of Employees", "Payment Mechanics",
                "Insurance Certificates", "Records Retention Notices",
                "Cure of Minor Breaches", "Force Majeure Definitions"]

_FILLERS = [
    "Each party shall act reasonably and in good faith under this Section.",
    "No waiver of any breach shall constitute a waiver of any subsequent breach.",
    "The parties acknowledge that damages may be an inadequate remedy hereunder.",
    "Any dispute arising under this Section shall follow the procedures in Article 14.",
    "This Section survives termination of the Agreement to the extent necessary.",
    "The exercise of rights hereunder is not exclusive of any other rights.",
]

_JURISDICTIONS = ["the State of Delaware", "New York", "the Commonwealth of Massachusetts",
                  "the Province of Ontario", "England and Wales"]


def _section_body(rng: random.Random, key: str, kind: str, base: date):
    """Returns (full_section_text, answer_registration) where
    answer_registration is (field_key, gold_text, kind) or None for pre-header
    filler paragraphs (answer lands BELOW the header sometimes)."""
    pre = [rng.choice(_FILLERS) for _ in range(rng.randint(0, 2))]
    if kind == _P:
        title = key.replace("_", " ").lower()
        ans = (f"Neither party may transfer or exercise rights regarding the {title} "
               f"without prior written consent of the other party, except that either "
               f"party may disclose the {title} to its professional advisors bound by "
               f"equivalent obligations, and any permitted transfer binds successors.")
        body = " ".join(pre + [ans])
        return body, (key, ans, _P)
    if kind == _M:
        amount = round(rng.uniform(500, 90_000), 2)
        text = format_money(rng, amount)
        noun = key.replace("_", " ")
        body = " ".join(pre + [f"Except as stated elsewhere, {noun} is agreed as follows: "
                               f"upon any qualifying event, a charge equal to {text} shall apply. "
                               f"The parties note the schedule in Exhibit B "
                               f"(reference {rng.randint(2015, 2025)}-{rng.randint(100, 999)})."])
        return body, (key, text, _M)
    if kind == _D:
        d = base + timedelta(days=rng.randint(90, 900))
        text = format_date(rng, d)
        noun = key.replace("_", " ")
        body = " ".join(pre + [f"The {noun} is fixed at {text}, unless extended in writing."])
        return body, (key, text, _D)
    if kind == _U:
        unit = "days" if key == "cure_period" else "months"
        n = rng.choice([10, 15, 30, 45, 60, 90]) if unit == "days" else rng.choice([6, 12, 24, 36])
        noun = key.replace("_", " ")
        body = " ".join(pre + [f"A party in breach shall have {n} {unit} to cure after written "
                               f"notice, and the {noun} shall not be shortened without consent."])
        return body, (key, f"{n} {unit}", _U)
    if kind == _L:
        addr = f"{rng.choice(COMPANIES)}, {rng.randint(100, 9999)} {rng.choice(['Industrial Way', 'Market Street', 'Harbor Blvd', 'Queen St W'])}"
        body = f"Notices under this Agreement: {addr}."
        return body, (key, addr, _L)
    raise ValueError(kind)


def contract(rng: random.Random, index: int) -> dict[str, Any]:
    builder = Builder()
    base = date(2026, rng.randint(1, 10), rng.randint(1, 27))
    client = rng.choice(COMPANIES)
    vendor = rng.choice([c for c in COMPANIES if c != client])
    builder.plain(f"MASTER SERVICES AGREEMENT\n\n")
    builder.value("contract_parties", f"between {client} and {vendor}")
    builder.plain(" dated ")
    builder.value("effective_date", format_date(rng, base))
    builder.plain(".\n\n")

    keys = rng.sample(list(CLAUSE_KINDS), k=rng.randint(7, 12))
    present = [k for k in keys if CLAUSE_KINDS[k] != _L or rng.random() < 0.7]
    rng.shuffle(present)
    fields: dict[str, Any] = {}
    answers: dict[str, Any] = {}
    addr_items: list[str] = []

    n = 0
    for key in present:
        kind = CLAUSE_KINDS[key]
        n += 1
        title_style = rng.choice(["title", "upper", "snake"])
        title = key.replace("_", " ")
        if title_style == "title":
            title = title.title()
        elif title_style == "upper":
            title = title.upper()
        else:
            title = key.upper()
        builder.plain(f"{n}. {title}\n")
        body, (ans_key, ans_text, ans_kind) = _section_body(rng, key, kind, base)
        if ans_kind == _L:
            builder.plain(body)
            builder.plain("\n\n")
            addr_items.append(ans_text)
            continue
        # register answer span inside the body
        pos = body.rfind(ans_text)
        assert pos >= 0
        head_part, tail = body[:pos], body[pos + len(ans_text):]
        parts = (head_part, ans_text, tail)
        for p_i, part in enumerate(parts):
            if part:
                if p_i == 1:
                    builder.value(ans_key, part)
                else:
                    builder.plain(part)
        builder.plain("\n\n")

    # decoy sections with near-miss titles (no registered answers)
    for dtitle in rng.sample(DECOY_TITLES, k=rng.randint(2, 5)):
        n += 1
        builder.plain(f"{n}. {dtitle}\n")
        builder.plain(" ".join(rng.sample(_FILLERS, k=2) +
                               [f"Fees under this Section accrue at {format_money(rng, round(rng.uniform(100, 5000), 2))} per event."]))
        builder.plain("\n\n")

    if "notice_addresses" in present and addr_items:
        # repeat + extra addresses in the closing section -> list<text> gold
        n += 1
        builder.plain(f"{n}. Notices for Service of Process\n")
        extra = (f"{rng.choice(COMPANIES)}, {rng.randint(100, 9999)} "
                 f"{rng.choice(['Broadway', 'King Ave', 'Cann Street'])}")
        items = (addr_items + [extra])[:3]
        for slot, item in enumerate(items):
            if slot:
                builder.plain("\n")
            builder.value("notice_addresses", item, item=True, index=slot)
        builder.plain("\n\n")
        answers["notice_addresses"] = {"value": list(items)}

    # governing jurisdiction text (short span, deep in final article)
    n += 1
    builder.plain(f"{n}. Governing Law\nThis Agreement is governed by the laws of ")
    builder.value("governing_law", rng.choice(_JURISDICTIONS))
    builder.plain(", without regard to conflict-of-laws principles.\n")

    state = builder.text()
    # assemble fields/answers from registered spans
    for key in present:
        kind = CLAUSE_KINDS[key]
        if key == "notice_addresses":
            continue
        t = {"paragraph": "text", "money": "number", "date": "date",
             "duration": "duration"}[kind]
        spec: dict[str, Any] = {"type": t, "instructions": _clause_instruction(rng, key)}
        if kind == _M:
            spec.update({"unit": "USD", "decimals": 2})
        if kind == _U:
            spec.update({"unit": "days" if key == "cure_period" else "months",
                         "integer": True})
        fields[key] = spec
        if key in builder.spans:
            s, e = builder.spans[key]
            gold_val = state[s:e]
            if kind == _M:
                answers[key] = {"value": parse_money_text(gold_val), "span": [s, e]}
            elif kind == _D:
                answers[key] = {"value": parse_date(gold_val), "span": [s, e]}
            elif kind == _U:
                answers[key] = {"value": float(gold_val.split()[0]), "span": [s, e]}
            else:
                answers[key] = {"value": gold_val, "span": [s, e]}
    fields["contract_parties"] = {"type": "text", "instructions": "the parties to this agreement"}
    fields["effective_date"] = {"type": "date", "instructions": "date this agreement takes effect"}
    fields["governing_law"] = {"type": "text", "instructions": "which law governs the agreement"}
    for k in ("contract_parties", "effective_date", "governing_law"):
        s, e = builder.spans[k]
        answers[k] = {"value": state[s:e], "span": [s, e]}
    answers["effective_date"]["value"] = base.isoformat()
    if addr_items:
        fields["notice_addresses"] = {"type": "list", "items": "text",
                                      "instructions": "addresses for official notices"}
    # absent fields: request things not in the document (~40% of schema)
    pool = [k for k in CLAUSE_KINDS if k not in fields]
    for key in rng.sample(pool, k=min(len(pool), rng.randint(8, 16))):
        kind = CLAUSE_KINDS[key]
        t = {"paragraph": "text", "money": "number", "date": "date",
             "duration": "duration", "list-addr": "list"}[kind]
        spec = {"type": t, "instructions": _clause_instruction(rng, key)}
        if kind == "money":
            spec.update({"unit": "USD", "decimals": 2})
        if kind == "list-addr":
            spec["items"] = "text"
        fields[key] = spec
        answers.setdefault(key, {"value": None})
    return make_record(builder, fields, answers, "contract", index, rng)


def parse_money_text(text: str) -> float | None:
    try:
        return float(text.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _clause_instruction(rng: random.Random, key: str) -> str:
    noun = key.replace("_", " ")
    return rng.choice([f"the clause text on {noun}", f"state the {noun} terms",
                       f"what does the agreement say about {noun}", f"{noun}, if present"])


DOMAINS = {
    "support": support_email,
    "invoice": invoice,
    "ticket": ticket,
    "table": table_record,
    "comps": comparables,
    "forecast": forecast_record,
    "contract": contract,
}

CUE_RE = re.compile(r"^([A-Za-z][A-Za-z /\-]{2,30}): ?(.+)$")

# Tier D targets prose domains where 'Label: value' lines teach the pointer
# to answer by adjacency. CSV blocks, item rows and forecast tables must
# keep their structure (aggregates and horizon semantics depend on it).
SCATTER_DOMAINS = ("invoice", "ticket")
SCATTER_PLACEHOLDER = "value appears later in this document."


def _line_table(state: str) -> tuple[list[tuple[int, str]], list[int]]:
    lines, starts, pos = [], [], 0
    for line in state.split("\n"):
        lines.append((pos, line))
        starts.append(pos)
        pos += len(line) + 1
    return lines, starts


def scatter_record(rng: random.Random, record: dict[str, Any]) -> dict[str, Any]:
    """Tier D: dissolve within-line label-adjacency at the data level.

    A 'Tax: $308.03' line becomes 'Tax: value appears later...' and the bare
    '$308.03' is planted >=2 lines below, so cue-to-value binding has to run
    through meaning, not position. Spans are remapped POSITIONALLY (we know
    exactly where every line's text lands) and every carried span is
    byte-verified against the rewritten state; a field whose span cannot be
    carried cleanly is dropped from fields+answers together, never corrupted
    (search-based re-derivation was rejected: money values like 308.03 vs
    '$308.03' and legitimately repeated atoms would silently mangle labels).
    """
    if record.get("domain") not in SCATTER_DOMAINS:
        return record
    state = record["state"]
    if state.count("\n") < 3:
        return record
    lines, starts = _line_table(state)

    def line_of(pos: int) -> int:
        return bisect.bisect_right(starts, pos) - 1

    # boundaries that fall inside a multi-line span: never insert there
    blocked: set[int] = set()
    all_spans = [tuple(v["span"]) for v in record["answers"].values()
                 if isinstance(v, dict) and v.get("span")]
    for answer in record["answers"].values():
        if isinstance(answer, dict) and answer.get("spans"):
            all_spans += [tuple(sp) for sp in answer["spans"]]
    for s, e in all_spans:
        for j in range(line_of(s), line_of(e - 1)):
            blocked.add(j)

    field_span = {k: tuple(v["span"]) for k, v in record["answers"].items()
                  if isinstance(v, dict) and v.get("span")}

    # cue lines whose value IS the whole line suffix are move candidates
    candidates: list[tuple[int, str, str | None, str]] = []  # (line, label, field|None, value)
    for idx, (base, line) in enumerate(lines):
        if line.count(",") >= 2:
            continue
        m = CUE_RE.match(line)
        if not m:
            continue
        value = line[m.start(2):]
        if not (1 < len(value) <= 44):
            continue
        hits = [f for f, (s, e) in field_span.items()
                if s == base + m.start(2) and e == base + len(line)]
        if hits:
            field: str | None = hits[0]
        else:
            # no gold span claims this line, but a label-adjacent atom
            # ("Subtotal: $4,248.73") is exactly what teaches the NUMERIC
            # pointer to answer by lexical adjacency — move the bare value too
            vs, ve = base + m.start(2), base + len(line)
            if any(s < ve and e > vs for s, e in field_span.values()):
                continue
            if len(value) > 30 or not any(ch.isdigit() for ch in value):
                continue
            field = None
        if rng.random() < 0.8:
            candidates.append((idx, m.group(1), field, value))
    if not candidates:
        return record

    # place each moved value >=2 lines under its cue, outside blocked spans
    rng.shuffle(candidates)
    placed: dict[int, list[tuple[str, str]]] = {}
    repl: dict[int, str] = {}
    moved: dict[str, str] = {}
    for idx, label, field, value in candidates:
        options = [j for j in range(idx + 2, len(lines)) if j not in blocked]
        if not options:
            continue  # nowhere clean to put it: leave this cue line intact
        key = field if field is not None else f"~atom-{idx}"  # atom moves are spanless
        placed.setdefault(rng.choice(options), []).append((key, value))
        repl[idx] = f"{label}: {SCATTER_PLACEHOLDER}"
        moved[key] = value
    if not moved:
        return record

    # rebuild the state, tracking where each line and each moved value lands
    parts: list[str] = []
    line_new_start: dict[int, int] = {}
    moved_start: dict[str, int] = {}
    pos = 0
    for j, (_, text) in enumerate(lines):
        line_new_start[j] = pos
        line_text = repl.get(j, text)
        parts.append(line_text)
        pos += len(line_text) + 1
        for field, value in placed.get(j, ()):
            moved_start[field] = pos
            parts.append(value)
            pos += len(value) + 1
    new_state = "\n".join(parts)

    def carry_span(s: int, e: int) -> tuple[int, int] | None:
        l0, l1 = line_of(s), line_of(e - 1)
        if any(j in repl for j in range(l0, l1 + 1)):
            return None  # this span touches a rewritten cue line
        return (line_new_start[l0] + (s - starts[l0]),
                line_new_start[l1] + (e - starts[l1]))

    for name in list(record["answers"]):
        answer = record["answers"][name]
        if not isinstance(answer, dict):
            continue
        ok = True
        if answer.get("span"):
            s, e = answer["span"][0], answer["span"][1]
            if name in moved_start:
                span = (moved_start[name], moved_start[name] + len(moved[name]))
            else:
                span = carry_span(s, e)
            if span is None or new_state[span[0]:span[1]] != state[s:e]:
                ok = False
            else:
                answer["span"] = [span[0], span[1]]
        if answer.get("spans") and ok:
            remapped = []
            for s, e in answer["spans"]:
                span = carry_span(s, e)
                if span is None or new_state[span[0]:span[1]] != state[s:e]:
                    ok = False
                    break
                remapped.append([span[0], span[1]])
            if ok:
                answer["spans"] = remapped
        if not ok:
            del record["answers"][name]
            record["fields"].pop(name, None)

    record["state"] = new_state
    return record


def generate(count: int, seed: int = 0, domains: list[str] | None = None,
             cue_scatter: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names = domains or list(DOMAINS)
    records = [DOMAINS[name](rng, i) for i, name in enumerate(rng.choices(names, k=count))]
    if cue_scatter:
        scatter_rng = random.Random(seed + 7919)  # independent stream: base data unchanged
        records = [scatter_record(scatter_rng, r) for r in records]
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic extraction records")
    parser.add_argument("--out", type=Path, required=True, help="output .jsonl path")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domains", nargs="*", default=None, choices=list(DOMAINS))
    parser.add_argument("--zip", action="store_true", help="also write a .zip next to the jsonl")
    parser.add_argument("--cue-scatter", dest="cue_scatter", action="store_true",
                        help="Tier D: move 'Label: value' figures >=2 lines off their cue line")
    args = parser.parse_args(argv)

    records = generate(args.count, args.seed, args.domains, cue_scatter=args.cue_scatter)
    write_jsonl(records, args.out)
    if args.zip:
        with zipfile.ZipFile(str(args.out) + ".zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(args.out, args.out.name)
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
