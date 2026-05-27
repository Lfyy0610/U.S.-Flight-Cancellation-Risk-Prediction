"""Part 4 monitoring and drift demonstration.

This script creates a deterministic data-drift demo from the 2023 test slice.
It intentionally uses only the Python standard library so the report can be
generated in lightweight environments without extra setup.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "flights_sample_3m.csv"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "drift_report.html"

CATEGORICAL_FEATURES = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "ROUTE",
    "DEP_PERIOD",
]

NUMERIC_FEATURES = [
    "DEP_HOUR",
    "DEP_MIN",
    "IS_WEEKEND",
    "IS_HOLIDAY_SEASON",
    "IS_SUMMER_PEAK",
    "IS_COVID_YEAR",
]

MONITORING_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def hour_to_period(hour: int) -> str:
    if hour < 6:
        return "red_eye"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def build_monitoring_row(raw: dict[str, str]) -> dict[str, object]:
    flight_date = datetime.strptime(raw["FL_DATE"], "%Y-%m-%d")
    crs_dep_time = parse_int(raw.get("CRS_DEP_TIME"))
    dep_hour = max(0, min(23, crs_dep_time // 100))
    dep_min = max(0, min(59, crs_dep_time % 100))
    airline = str(raw.get("AIRLINE_CODE", "")).strip().upper()
    origin = str(raw.get("ORIGIN", "")).strip().upper()
    dest = str(raw.get("DEST", "")).strip().upper()

    return {
        "YEAR": flight_date.year,
        "MONTH": flight_date.month,
        "DAY_OF_WEEK": flight_date.weekday(),
        "CANCELLED": parse_int(raw.get("CANCELLED")),
        "AIRLINE_CODE": airline,
        "ORIGIN": origin,
        "DEST": dest,
        "ROUTE": f"{origin}_{dest}",
        "DEP_HOUR": dep_hour,
        "DEP_MIN": dep_min,
        "DEP_PERIOD": hour_to_period(dep_hour),
        "IS_WEEKEND": int(flight_date.weekday() in (5, 6)),
        "IS_HOLIDAY_SEASON": int(flight_date.month in (11, 12, 1)),
        "IS_SUMMER_PEAK": int(flight_date.month in (6, 7, 8)),
        "IS_COVID_YEAR": int(flight_date.year == 2020),
    }


def load_reference_rows(csv_path: Path, sample_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                row = build_monitoring_row(raw)
            except (KeyError, ValueError):
                continue
            if row["YEAR"] != 2023:
                continue
            rows.append(row)
            if len(rows) >= sample_size:
                break
    if not rows:
        raise RuntimeError(f"No 2023 monitoring rows found in {csv_path}")
    return rows


def create_modified_current_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    current = deepcopy(rows)
    n = len(current)
    if n == 0:
        return current

    for idx, row in enumerate(current):
        if idx < int(n * 0.40):
            row["AIRLINE_CODE"] = "WN"
        if idx < int(n * 0.30):
            row["ORIGIN"] = "LAX"
        if idx < int(n * 0.25):
            row["DEP_HOUR"] = 23
            row["DEP_MIN"] = 45
            row["DEP_PERIOD"] = "red_eye"
        if idx < int(n * 0.20):
            row["IS_HOLIDAY_SEASON"] = 1
        row["ROUTE"] = f"{row['ORIGIN']}_{row['DEST']}"
    return current


def categorical_psi(reference: Iterable[object], current: Iterable[object]) -> float:
    ref_counts = Counter(reference)
    cur_counts = Counter(current)
    ref_total = sum(ref_counts.values())
    cur_total = sum(cur_counts.values())
    if ref_total == 0 or cur_total == 0:
        return 0.0

    score = 0.0
    for value in set(ref_counts) | set(cur_counts):
        ref_pct = max(ref_counts[value] / ref_total, 1e-6)
        cur_pct = max(cur_counts[value] / cur_total, 1e-6)
        score += (cur_pct - ref_pct) * math.log(cur_pct / ref_pct)
    return score


def numeric_psi(reference: list[object], current: list[object]) -> float:
    ref_values = [float(x) for x in reference]
    cur_values = [float(x) for x in current]
    unique_values = sorted(set(ref_values + cur_values))
    if len(unique_values) <= 10:
        return categorical_psi(ref_values, cur_values)

    min_value = min(ref_values)
    max_value = max(ref_values)
    if min_value == max_value:
        return 0.0

    bins = 10
    width = (max_value - min_value) / bins

    def bucket(value: float) -> int:
        if value >= max_value:
            return bins - 1
        return max(0, min(bins - 1, int((value - min_value) / width)))

    return categorical_psi([bucket(v) for v in ref_values], [bucket(v) for v in cur_values])


def top_distribution(rows: list[dict[str, object]], feature: str, limit: int = 5) -> list[tuple[str, float]]:
    counts = Counter(str(row[feature]) for row in rows)
    total = sum(counts.values()) or 1
    return [(value, count / total * 100) for value, count in counts.most_common(limit)]


def drift_level(score: float) -> str:
    if score >= 0.25:
        return "High"
    if score >= 0.10:
        return "Moderate"
    return "Low"


def compute_drift(reference: list[dict[str, object]], current: list[dict[str, object]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for feature in MONITORING_FEATURES:
        ref_values = [row[feature] for row in reference]
        cur_values = [row[feature] for row in current]
        score = (
            categorical_psi(ref_values, cur_values)
            if feature in CATEGORICAL_FEATURES
            else numeric_psi(ref_values, cur_values)
        )
        results.append(
            {
                "feature": feature,
                "psi": score,
                "level": drift_level(score),
                "reference_top": top_distribution(reference, feature),
                "current_top": top_distribution(current, feature),
            }
        )
    return sorted(results, key=lambda item: float(item["psi"]), reverse=True)


def format_distribution(items: list[tuple[str, float]]) -> str:
    return "<br>".join(f"{html.escape(value)}: {pct:.1f}%" for value, pct in items)


def render_report(reference: list[dict[str, object]], current: list[dict[str, object]], results: list[dict[str, object]]) -> str:
    high_count = sum(1 for row in results if row["level"] == "High")
    moderate_count = sum(1 for row in results if row["level"] == "Moderate")
    rows_html = "\n".join(
        f"""
        <tr class="{str(result['level']).lower()}">
          <td>{html.escape(str(result['feature']))}</td>
          <td>{float(result['psi']):.4f}</td>
          <td>{html.escape(str(result['level']))}</td>
          <td>{format_distribution(result['reference_top'])}</td>
          <td>{format_distribution(result['current_top'])}</td>
        </tr>
        """
        for result in results
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Flight Cancellation Drift Monitoring Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d1d5db; border-radius: 6px; padding: 14px; background: #f9fafb; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #e5e7eb; }}
    tr.high td {{ background: #fee2e2; }}
    tr.moderate td {{ background: #fef3c7; }}
    tr.low td {{ background: #ecfdf5; }}
    .note {{ background: #eff6ff; border-left: 4px solid #2563eb; padding: 12px; margin: 18px 0; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Flight Cancellation Drift Monitoring Report</h1>
  <p>Reference data is the original 2023 test slice. Current data is a deterministic modified version used to simulate production drift.</p>

  <div class="summary">
    <div class="metric">Reference rows<strong>{len(reference):,}</strong></div>
    <div class="metric">Current rows<strong>{len(current):,}</strong></div>
    <div class="metric">High drift features<strong>{high_count}</strong></div>
    <div class="metric">Moderate drift features<strong>{moderate_count}</strong></div>
  </div>

  <div class="note">
    PSI interpretation: below 0.10 is low drift, 0.10 to 0.25 is moderate drift, and 0.25 or above is high drift.
  </div>

  <h2>Feature Drift Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Feature</th>
        <th>PSI</th>
        <th>Drift Level</th>
        <th>Original Test Distribution</th>
        <th>Modified Test Distribution</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <h2>Presentation Interpretation</h2>
  <p>The modified test data intentionally shifts carrier, origin airport, route, departure period, and holiday-season features. These are the same feature families used by the deployed model, so drift in these inputs can change the reliability of cancellation-risk predictions.</p>
</body>
</html>
"""


def write_report(report: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Part 4 flight cancellation drift report.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to flights_sample_3m.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH, help="HTML report output path")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of 2023 rows to sample")
    args = parser.parse_args()

    reference = load_reference_rows(args.data, args.sample_size)
    current = create_modified_current_rows(reference)
    results = compute_drift(reference, current)
    report = render_report(reference, current, results)
    write_report(report, args.output)

    print(f"Wrote drift report: {args.output}")
    for result in results[:5]:
        print(f"{result['feature']}: PSI={float(result['psi']):.4f}, level={result['level']}")


if __name__ == "__main__":
    main()
