"""Hand-written SVG charts.

Each function returns an SVG string whose marks use CSS classes rather than literal colours, so
one chart serves the light and dark page and, with `standalone`, the README. Marks that deserve
a tooltip carry `data-tip`; the page attaches one small hover handler to all of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape
from pathlib import Path

from . import config

ASSETS = Path(__file__).parent / "assets"
PLOT_INSET = 32
MINUS = "\u2212"


@dataclass(frozen=True)
class Linear:
    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, value: float) -> float:
        return self.r0 + (value - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = (high - low) or 1.0
    raw = span / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(low / step)
    ticks = []
    value = first * step
    while value <= high + 1e-9:
        ticks.append(round(value, 6))
        value += step
    return ticks


def fmt_pct(value: float, digits: int = 1, sign: bool = True) -> str:
    text = f"{abs(value):.{digits}f}%"
    if value < 0 and round(abs(value), digits) != 0:
        return MINUS + text
    return ("+" + text) if sign and round(value, digits) != 0 else text


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def date_of_week(k: float) -> date:
    return config.POLICY_START + timedelta(days=7 * k)


def svg_open(width: int, height: int, title: str, description: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}: {escape(description)}" xmlns="http://www.w3.org/2000/svg">'
    )


def poly(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def year_axis(x: Linear, y_bottom: float, top: float, years: Sequence[int]) -> str:
    parts = []
    for year in years:
        k = config.week_index(date(year, 1, 1))
        if x.d0 <= k <= x.d1:
            px = x(k)
            parts.append(
                f'<line class="grid" x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{y_bottom}"/>'
            )
            parts.append(f'<text x="{px + 4:.1f}" y="{y_bottom + 16}">{year}</text>')
    return "".join(parts)


def y_axis(
    y: Linear,
    left: float,
    right: float,
    ticks: Sequence[float],
    suffix: str = "%",
    signed: bool = True,
) -> str:
    parts = []
    for tick in ticks:
        py = y(tick)
        cls = "zero" if tick == 0 else "grid"
        number = f"{tick:+g}" if signed else f"{tick:g}"
        label = f"{number}{suffix}".replace("-", MINUS) if tick else f"0{suffix}"
        parts.append(f'<line class="{cls}" x1="{left}" x2="{right}" y1="{py:.1f}" y2="{py:.1f}"/>')
        parts.append(f'<text x="{left}" y="{py - 5:.1f}">{label}</text>')
    return "".join(parts)


def start_marker(x: Linear, top: float, bottom: float, text: str = "Tolling starts") -> str:
    px = x(0)
    return (
        f'<line class="start" x1="{px:.1f}" x2="{px:.1f}" y1="{top}" y2="{bottom}"/>'
        f'<text class="note" x="{px + 6:.1f}" y="{top + 10}">{escape(text)}</text>'
    )


def event_study_chart(
    core: list[dict],
    title: str,
    description: str,
    ymin: float,
    ymax: float,
    height: int = 380,
    width: int = 760,
) -> str:
    left, right, top, bottom = 8, width - 8, 26, height - 28
    x = Linear(core[0]["week"], core[-1]["week"] + 4, left + PLOT_INSET, right)
    y = Linear(ymin, ymax, bottom, top)
    out = [svg_open(width, height, title, description)]
    out.append(y_axis(y, left, right, nice_ticks(ymin, ymax, 6)))
    years = sorted({int(r["start"][:4]) for r in core})
    out.append(year_axis(x, bottom, top, [*years, years[-1] + 1]))

    split = [r for r in core if r["week"] < -52], [r for r in core if r["week"] >= 0]
    for rows in split:
        if not rows:
            continue
        upper = [(x(r["week"] + 2), y(r["high"])) for r in rows]
        lower = [(x(r["week"] + 2), y(r["low"])) for r in reversed(rows)]
        out.append(f'<polygon class="band-blue" points="{poly(upper + lower)}"/>')
        path = [(x(r["week"] + 2), y(r["percent"])) for r in rows]
        out.append(f'<polyline class="line-blue" points="{poly(path)}"/>')
        for r in rows:
            interval = f"{fmt_pct(r['low'])} to {fmt_pct(r['high'])}"
            tip = f"{r['start']}: {fmt_pct(r['percent'])} ({interval})"
            out.append(
                f'<rect class="hit" x="{x(r["week"]):.1f}" y="{top}" '
                f'width="{x(r["week"] + 4) - x(r["week"]):.1f}" height="{bottom - top}" '
                f'data-tip="{escape(tip)}"/>'
            )

    gap_left, gap_right = x(-52), x(0)
    middle = (gap_left + gap_right) / 2
    out.append(
        f'<text class="note" text-anchor="middle" x="{middle:.1f}" y="{y(0) - 8:.1f}">'
        f"{'2024: reference' if width < 600 else '2024 is the reference year'}</text>"
    )
    out.append(start_marker(x, top, bottom))
    out.append("</svg>")
    return "".join(out)


def series_chart(
    weeks: Sequence[int],
    series: dict[str, tuple[list[float], str]],
    title: str,
    description: str,
    ymin: float,
    ymax: float,
    ghosts: Sequence[Sequence[float]] = (),
    height: int = 340,
    suffix: str = "%",
    label_at: dict[str, tuple[float, float]] | None = None,
    width: int = 760,
    signed_axis: bool = True,
) -> str:
    """Lines against weeks. `series` maps a direct label to (values, css colour name)."""
    left, right, top, bottom = 8, width - 8, 26, height - 28
    x = Linear(weeks[0], weeks[-1], left + PLOT_INSET, right)
    y = Linear(ymin, ymax, bottom, top)
    out = [svg_open(width, height, title, description)]
    out.append(y_axis(y, left, right, nice_ticks(ymin, ymax, 6), suffix, signed_axis))
    years = range(date_of_week(weeks[0]).year, date_of_week(weeks[-1]).year + 2)
    out.append(year_axis(x, bottom, top, list(years)))
    for ghost in ghosts:
        path = [(x(k), y(min(max(v, ymin), ymax))) for k, v in zip(weeks, ghost, strict=True)]
        out.append(f'<polyline class="line-ghost" points="{poly(path)}"/>')
    for name, (values, colour) in series.items():
        path = [(x(k), y(v)) for k, v in zip(weeks, values, strict=True) if v is not None]
        out.append(f'<polyline class="line-{colour}" points="{poly(path)}"/>')
        if label_at and name in label_at:
            k, v = label_at[name]
            out.append(f'<text class="label" x="{x(k):.1f}" y="{y(v):.1f}">{escape(name)}</text>')
    step = max(1, len(weeks) // 96)
    names = list(series)
    for i in range(0, len(weeks), step):
        k = weeks[i]
        lines = [f"{date_of_week(k)}"]
        for name in names:
            value = series[name][0][i]
            if value is not None:
                lines.append(f"{name}: {fmt_pct(value, sign=signed_axis)}")
        out.append(
            f'<rect class="hit" x="{x(k):.1f}" y="{top}" '
            f'width="{(right - left) / len(weeks) * step + 1:.1f}" height="{bottom - top}" '
            f'data-tip="{escape(" | ".join(lines))}"/>'
        )
    out.append(start_marker(x, top, bottom))
    out.append("</svg>")
    return "".join(out)


def interval_chart(
    rows: list[dict],
    title: str,
    description: str,
    xmin: float,
    xmax: float,
    label_width: int = 250,
    row_height: int = 27,
    width: int = 760,
) -> str:
    """Dot and interval rows. Keys: label, value, low, high, optional section and emphasis.

    Below 600 units there is no room for a label column, so each label sits above its interval.
    """
    stacked = width < 600
    row_height = 44 if stacked else row_height
    sections = []
    for row in rows:
        section = row.get("section")
        if section and section not in sections:
            sections.append(section)
    height = 34 + row_height * len(rows) + 30 * len(sections) + 26
    left, right = (12, width - 12) if stacked else (label_width, width - 16)
    x = Linear(xmin, xmax, left, right)
    bottom = height - 26
    out = [svg_open(width, height, title, description)]
    for tick in nice_ticks(xmin, xmax, 4 if stacked else 6):
        px = x(tick)
        cls = "zero" if tick == 0 else "grid"
        label = f"{tick:+g}%".replace("-", MINUS) if tick else "0%"
        out.append(f'<line class="{cls}" x1="{px:.1f}" x2="{px:.1f}" y1="10" y2="{bottom}"/>')
        out.append(f'<text text-anchor="middle" x="{px:.1f}" y="{bottom + 16}">{label}</text>')
    cursor = 14.0
    current = None
    for row in rows:
        if row.get("section") != current:
            current = row.get("section")
            if current:
                cursor += 16
                out.append(f'<text class="note" x="0" y="{cursor:.1f}">{escape(current)}</text>')
                cursor += 12
        strong = row.get("emphasis")
        tone = "blue" if strong else "ink"
        cls = "label" if strong else ""
        cy = cursor + (31 if stacked else row_height / 2)
        label_y = cursor + 14 if stacked else cy + 4
        out.append(f'<text class="{cls}" x="0" y="{label_y:.1f}">{escape(row["label"])}</text>')
        clipped_low = max(row["low"], xmin) if row.get("low") is not None else None
        clipped_high = min(row["high"], xmax) if row.get("high") is not None else None
        tip = f"{row['label']}: {fmt_pct(row['value'])}"
        if clipped_low is not None and clipped_high is not None:
            tip += f" ({fmt_pct(row['low'])} to {fmt_pct(row['high'])})"
            out.append(
                f'<line class="bar-{tone}" x1="{x(clipped_low):.1f}" x2="{x(clipped_high):.1f}" '
                f'y1="{cy:.1f}" y2="{cy:.1f}"/>'
            )
        out.append(f'<circle class="dot-{tone}" cx="{x(row["value"]):.1f}" cy="{cy:.1f}" r="5.5"/>')
        out.append(
            f'<rect class="hit" x="0" y="{cursor:.1f}" width="{width}" height="{row_height}" '
            f'data-tip="{escape(tip)}"/>'
        )
        cursor += row_height
    out.append("</svg>")
    return "".join(out)


def strip_chart(
    placebos: list[float],
    actual: float,
    title: str,
    description: str,
    xmin: float,
    xmax: float,
    actual_label: str,
    placebo_label: str,
    width: int = 760,
) -> str:
    """One row of grey placebo dots beside the highlighted real estimate."""
    height, left, right = 150, 14, width - 14
    x = Linear(xmin, xmax, left, right)
    baseline = 86
    out = [svg_open(width, height, title, description)]
    for tick in nice_ticks(xmin, xmax, 6):
        px = x(tick)
        cls = "zero" if tick == 0 else "grid"
        label = f"{tick:+g}%".replace("-", MINUS) if tick else "0%"
        out.append(f'<line class="{cls}" x1="{px:.1f}" x2="{px:.1f}" y1="44" y2="118"/>')
        out.append(f'<text text-anchor="middle" x="{px:.1f}" y="136">{label}</text>')
    taken: list[tuple[float, int]] = []
    for value in sorted(placebos):
        px = x(min(max(value, xmin), xmax))
        level = 0
        while any(abs(px - p) < 11 and lv == level for p, lv in taken):
            level += 1
        taken.append((px, level))
        cy = baseline - level * 11
        out.append(
            f'<circle class="dot-ghost" cx="{px:.1f}" cy="{cy}" r="5.5" '
            f'data-tip="{escape(placebo_label)}: {fmt_pct(value)}"/>'
        )
    px = x(actual)
    anchor = "end" if px > width * 0.7 else "start"
    shift = -10 if anchor == "end" else 10
    out.append(
        f'<circle class="dot-blue" cx="{px:.1f}" cy="{baseline + 18}" r="7" '
        f'data-tip="{escape(actual_label)}: {fmt_pct(actual)}"/>'
    )
    out.append(
        f'<text class="label" text-anchor="{anchor}" x="{px + shift:.1f}" y="{baseline + 23}">'
        f"{escape(actual_label)}</text>"
    )
    out.append(f'<text class="note" x="{left}" y="20">{escape(placebo_label)}</text>')
    out.append("</svg>")
    return "".join(out)


def map_chart(
    features: list[dict],
    classes: dict[int, str],
    tips: dict[int, str],
    title: str,
    description: str,
    labels: Sequence[tuple[float, float, str]] = (),
) -> str:
    """A zone map cropped to the area around Manhattan, in a local equirectangular projection."""
    west, east, south, north = -74.06, -73.84, 40.675, 40.835
    width = 560
    scale = width / ((east - west) * 0.7585)
    height = round((north - south) * scale)

    def px(lon: float, lat: float) -> tuple[float, float]:
        return (lon - west) * 0.7585 * scale, (north - lat) * scale

    def rings_of(geometry: dict) -> list[list]:
        if geometry["type"] == "Polygon":
            return geometry["coordinates"]
        return [ring for polygon in geometry["coordinates"] for ring in polygon]

    def path(geometry: dict) -> str:
        chunks = []
        for ring in rings_of(geometry):
            points = [px(lon, lat) for lon, lat in ring]
            chunks.append("M" + "L".join(f"{a:.1f} {b:.1f}" for a, b in points) + "Z")
        return "".join(chunks)

    out = [svg_open(width, height, title, description)]
    out.append(
        f'<defs><clipPath id="crop"><rect width="{width}" height="{height}"/></clipPath></defs>'
    )
    out.append('<g clip-path="url(#crop)">')
    toll = None
    for feature in features:
        location = feature["properties"]["location_id"]
        if location == 0:
            toll = feature
            continue
        cls = classes.get(location, "z-far")
        tip = escape(tips.get(location, ""))
        out.append(f'<path class="zone {cls}" d="{path(feature["geometry"])}" data-tip="{tip}"/>')
    if toll is not None:
        out.append(f'<path class="toll-line" d="{path(toll["geometry"])}"/>')
    out.append("</g>")
    for lon, lat, text in labels:
        a, b = px(lon, lat)
        out.append(f'<text class="label" x="{a:.1f}" y="{b:.1f}">{escape(text)}</text>')
    out.append("</svg>")
    return "".join(out)


def standalone(svg: str, background: str = "#fcfcfb") -> str:
    """Inline the styles so a chart renders on its own, as in the README."""
    css = (ASSETS / "chart.css").read_text()
    style = f"<style>{css}svg.chart{{background:{background}}}</style>"
    head, rest = svg.split(">", 1)
    return f"{head}>{style}{rest}"
