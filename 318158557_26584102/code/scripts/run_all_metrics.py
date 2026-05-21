import argparse
import subprocess
import sys
import time


METRIC_MODULES = {
    "ssim": "src.eval_ssim_celebhq",
    "lpips": "src.eval_lpips_celebhq",
    "psnr": "src.eval_psnr_celebhq",
    "fid": "src.eval_fid_celebhq",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all CelebHQ evaluation metrics sequentially."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_MODULES.keys(),
        default=list(METRIC_MODULES.keys()),
        help="Metrics to run. Defaults to all metrics.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining metrics even if one metric fails.",
    )
    return parser.parse_args()


def run_metric(metric_name: str) -> int:
    module = METRIC_MODULES[metric_name]
    print(f"\n=== Running {metric_name.upper()} ({module}) ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, "-m", module], check=False)
    elapsed_min = (time.time() - start) / 60.0
    status = "finished" if result.returncode == 0 else "failed"
    print(
        f"=== {metric_name.upper()} {status} in {elapsed_min:.1f} min ===",
        flush=True,
    )
    return result.returncode


def main():
    args = parse_args()
    failures = []

    for metric_name in args.metrics:
        returncode = run_metric(metric_name)
        if returncode != 0:
            failures.append((metric_name, returncode))
            if not args.continue_on_error:
                break

    if failures:
        summary = ", ".join(
            f"{metric} exit={returncode}" for metric, returncode in failures
        )
        raise SystemExit(f"Metric run failed: {summary}")

    print("\nAll requested metrics completed.", flush=True)


if __name__ == "__main__":
    main()
