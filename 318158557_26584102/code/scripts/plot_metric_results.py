import argparse
from collections import defaultdict
from pathlib import Path

from analyze_metric_results import DEFAULT_INPUTS, add_ranks, load_metric_csv, summarize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create bar plots from eval metric CSV outputs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Metric CSV files to plot.",
    )
    parser.add_argument(
        "--out-dir",
        default="metric_analysis/plots",
        help="Directory for generated plot images.",
    )
    return parser.parse_args()


def load_inputs(paths: list[str]) -> list[dict[str, object]]:
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            rows.extend(load_metric_csv(path))
    if not rows:
        raise SystemExit("No metric CSVs found to plot.")
    return rows


def group_ranked_rows(rows: list[dict[str, object]]):
    groups = defaultdict(list)
    for row in rows:
        key = (row["metric"], row["region"], row["mask_type"])
        groups[key].append(row)
    return groups


def plot_group(plt, group: list[dict[str, object]], out_dir: Path) -> Path:
    group = sorted(group, key=lambda row: int(row["rank"]))
    metric = str(group[0]["metric"])
    region = str(group[0]["region"])
    mask_type = str(group[0]["mask_type"])
    methods = [str(row["method"]) for row in group]
    means = [float(row["mean"]) for row in group]
    stds = [float(row["std"]) for row in group]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, means, yerr=stds, capsize=4)
    ax.set_title(f"{metric.upper()} {region} - {mask_type}")
    ax.set_xlabel("method")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = out_dir / f"{metric}_{region}_{mask_type}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install the project environment first."
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_rows = add_ranks(summarize(load_inputs(args.inputs)))
    paths = []
    for group in group_ranked_rows(ranked_rows).values():
        paths.append(plot_group(plt, group, out_dir))

    print(f"Saved {len(paths)} plots to {out_dir}")


if __name__ == "__main__":
    main()
