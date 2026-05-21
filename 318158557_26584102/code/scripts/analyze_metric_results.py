import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRIC_DIRECTIONS = {
    "ssim": "higher",
    "psnr": "higher",
    "lpips": "lower",
    "fid": "lower",
}

DEFAULT_INPUTS = [
    "eval_ssim_results.csv",
    "eval_lpips_results.csv",
    "eval_psnr_results.csv",
    "eval_fid_results.csv",
]

LONG_FIELDNAMES = ["idx", "mask_type", "metric", "region", "method", "value"]
SUMMARY_FIELDNAMES = [
    "metric",
    "region",
    "mask_type",
    "method",
    "n",
    "mean",
    "std",
    "median",
    "min",
    "max",
]
RANKED_FIELDNAMES = SUMMARY_FIELDNAMES + ["rank"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize eval metric CSVs and rank methods by metric/mask/region."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Metric CSV files to analyze.",
    )
    parser.add_argument(
        "--out-dir",
        default="metric_analysis",
        help="Directory for summary CSV outputs.",
    )
    return parser.parse_args()


def metric_from_path(path: Path) -> str:
    name = path.stem.lower()
    for metric in METRIC_DIRECTIONS:
        if metric in name:
            return metric
    raise ValueError(f"Could not infer metric name from {path}")


def parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Expected finite metric value, got {value}")
    return parsed


def wide_metric_to_long(rows: list[dict[str, str]], metric: str) -> list[dict[str, object]]:
    long_rows = []
    prefix = f"{metric}_"
    for row in rows:
        for col, raw_value in row.items():
            if col in {"idx", "mask_type"} or not col.startswith(prefix):
                continue
            region, method = col[len(prefix) :].rsplit("_", 1)
            long_rows.append(
                {
                    "idx": row.get("idx", ""),
                    "mask_type": row["mask_type"],
                    "metric": metric,
                    "region": region,
                    "method": method,
                    "value": parse_float(raw_value),
                }
            )
    if not long_rows:
        raise ValueError(f"No {metric} columns found")
    return long_rows


def fid_to_long(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    return [
        {
            "idx": "",
            "mask_type": row["mask_type"],
            "metric": "fid",
            "region": "all",
            "method": row["method"],
            "value": parse_float(row["fid"]),
        }
        for row in rows
    ]


def load_metric_csv(path: Path) -> list[dict[str, object]]:
    metric = metric_from_path(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if metric == "fid":
        return fid_to_long(rows)
    return wide_metric_to_long(rows, metric)


def summarize(long_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in long_rows:
        key = (row["metric"], row["region"], row["mask_type"], row["method"])
        groups[key].append(float(row["value"]))

    summary_rows = []
    for (metric, region, mask_type, method), values in groups.items():
        summary_rows.append(
            {
                "metric": metric,
                "region": region,
                "mask_type": mask_type,
                "method": method,
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        )
    return sorted(summary_rows, key=lambda r: (r["metric"], r["region"], r["mask_type"], r["method"]))


def add_ranks(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in summary_rows:
        key = (row["metric"], row["region"], row["mask_type"])
        groups[key].append(row)

    ranked_rows = []
    for (metric, _region, _mask_type), rows in groups.items():
        ascending = METRIC_DIRECTIONS[str(metric)] == "lower"
        sorted_rows = sorted(rows, key=lambda r: float(r["mean"]), reverse=not ascending)
        for rank, row in enumerate(sorted_rows, start=1):
            ranked_rows.append({**row, "rank": rank})
    return sorted(ranked_rows, key=lambda r: (r["metric"], r["region"], r["mask_type"], r["rank"]))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_best_methods(ranked_rows: list[dict[str, object]]) -> None:
    print("\nBest method by metric / region / mask:")
    for row in ranked_rows:
        if row["rank"] != 1:
            continue
        print(
            f"- {row['metric']} {row['region']} {row['mask_type']}: "
            f"{row['method']} mean={float(row['mean']):.6g}"
        )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []
    skipped = []
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.exists():
            skipped.append(path)
            continue
        long_rows.extend(load_metric_csv(path))

    if not long_rows:
        missing = ", ".join(str(path) for path in skipped)
        raise SystemExit(f"No metric CSVs found. Missing: {missing}")

    summary_rows = summarize(long_rows)
    ranked_rows = add_ranks(summary_rows)

    long_path = out_dir / "metrics_long.csv"
    summary_path = out_dir / "metrics_summary.csv"
    ranked_path = out_dir / "metrics_ranked.csv"
    write_csv(long_path, long_rows, LONG_FIELDNAMES)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDNAMES)
    write_csv(ranked_path, ranked_rows, RANKED_FIELDNAMES)

    print(f"Saved: {long_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {ranked_path}")
    if skipped:
        print("Skipped missing files:", ", ".join(str(path) for path in skipped))
    print_best_methods(ranked_rows)


if __name__ == "__main__":
    main()
