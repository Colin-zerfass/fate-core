"""
Parallel runner to execute Dynamical_cmems.py for each month in parallel.

Usage example:
    python run_months_parallel.py --workers 4 --out-dir ../output_parallel

This script launches independent Python subprocesses (one per month) using
`sys.executable` so each worker creates its own Parcels/Numba state safely.

It sets NUMBA_NUM_THREADS and OMP_NUM_THREADS to 1 in each worker's env to
avoid native thread oversubscription.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import time
import csv


def make_month_list(start_iso="2024-01-01", end_iso="2025-01-01"):
    # matches the logic in Dynamical_cmems.py
    monthrange = pd.date_range(start_iso, end_iso, freq="MS")
    months = []
    for month in range(len(monthrange) - 1):
        startdate = monthrange[month]
        enddate = monthrange[month + 1] + pd.Timedelta(days=7)
        months.append((month, str(startdate.date()), str(enddate.date())))
    return months


def run_month_process(script_path, month_index, out_dir, env_overrides, timeout=None):
    cmd = [sys.executable, str(script_path), "--month-index", str(month_index), "--out-dir", str(out_dir)]
    env = os.environ.copy()
    env.update(env_overrides)
    started = time.time()
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        rc = proc.returncode
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
    except Exception as e:
        rc = -1
        stdout = ""
        stderr = f"Exception while launching subprocess: {e}\n"
    ended = time.time()
    runtime = ended - started
    return {
        "month_index": month_index,
        "returncode": rc,
        "runtime_s": runtime,
        "stdout": stdout,
        "stderr": stderr,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Dynamical_cmems.py for each month in parallel")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1), help="Number of parallel subprocesses")
    parser.add_argument("--script", type=str, default="Code/Parcels/Dynamical Model/Dynamical_cmems.py", help="Path to Dynamical_cmems.py")
    parser.add_argument("--out-dir", type=str, default="Code/Parcels/output_parallel", help="Directory for per-month outputs and logs")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Month-range start (ISO YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01", help="Month-range end (ISO YYYY-MM-DD)")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue launching jobs even if some fail")
    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    months = make_month_list(args.start, args.end)
    print(f"Launching {len(months)} month jobs with up to {args.workers} workers")

    env_overrides = {
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    results = []

    with ThreadPoolExecutor(max_workers=min(args.workers, len(months))) as ex:
        future_to_month = {}
        for month_idx, _, _ in months:
            future = ex.submit(run_month_process, script_path, month_idx, out_dir, env_overrides)
            future_to_month[future] = month_idx

        for fut in as_completed(future_to_month):
            month_idx = future_to_month[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"month_index": month_idx, "returncode": -1, "runtime_s": 0, "stdout": "", "stderr": f"Executor exception: {e}"}
            results.append(res)
            # write per-job log
            log_path = out_dir / f"month_{res['month_index']}.log"
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("--- STDOUT ---\n")
                fh.write(res.get("stdout", ""))
                fh.write("\n--- STDERR ---\n")
                fh.write(res.get("stderr", ""))
            status = "OK" if res["returncode"] == 0 else f"FAIL({res['returncode']})"
            print(f"Month {res['month_index']}: {status}, runtime {res['runtime_s']:.1f}s, log={log_path}")
            if res["returncode"] != 0 and not args.continue_on_error:
                print("Aborting remaining jobs due to failure (use --continue-on-error to override)")
                # Cancel remaining futures
                for f in future_to_month:
                    if not f.done():
                        f.cancel()
                break

    # write summary CSV
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        fieldnames = ["month_index", "returncode", "runtime_s"]
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["month_index"]):
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
