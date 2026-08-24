#!/usr/bin/env python3
"""Probe SLURM for the (partition, GPUs, TP) combo that starts soonest.

Iterates over hardcoded candidate sizings that fit the model's bf16 weights,
runs `sbatch --test-only` on each to get SLURM's predicted start time, and
prints the winner as KEY=VAL lines on stdout (suitable for `eval $(...)`).

Sample diagnostic lines go to stderr.

Example:
    eval "$(scripts/pick_slot.py --model Qwen/Qwen3.6-27B --account <ACCOUNT>)"
    sbatch --partition=$PARTITION --gres=gpu:$GPUS --account=$ACCOUNT \\
           --export=ALL,TP=$TP,MODEL=...,CONDITION=... scripts/run_local_classify.slurm
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys

CANDIDATES = [
    ("<PARTITION_A100_80GB>", 1, 1, 80),
    ("<PARTITION_A100_80GB>", 2, 2, 80),
    ("<PARTITION_A100_80GB>", 4, 4, 80),
    ("<PARTITION_A100_40GB>", 2, 2, 40),
    ("<PARTITION_A100_40GB>", 4, 4, 40),
    ("<PARTITION_A100>",    2, 2, 40),
    ("<PARTITION_A100>",    4, 4, 40),
]


def model_weight_gb(model: str) -> float:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(model, files_metadata=True)
    total = 0
    for s in info.siblings:
        if s.rfilename.endswith(".safetensors") and s.size:
            total += s.size
    return total / 1e9


def fits(model_gb: float, tp: int, per_gpu_gb: int, util: float) -> bool:
    weight_per_gpu = model_gb / tp
    overhead = 4
    available = per_gpu_gb * util
    return weight_per_gpu + overhead <= available


def parse_test_only_start(stderr: str) -> dt.datetime | None:
    m = re.search(r"to start at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", stderr)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")


def probe(partition: str, gpus: int, account: str, time: str, mem: str,
          cpus: int) -> tuple[dt.datetime | None, str]:
    cmd = [
        "sbatch", "--test-only",
        "--cluster=<CLUSTER>",
        f"--partition={partition}",
        f"--account={account}",
        f"--gres=gpu:{gpus}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
        "--wrap", "true",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 and "Job" not in res.stderr:
        return None, res.stderr.strip().splitlines()[-1] if res.stderr else "unknown"
    ts = parse_test_only_start(res.stderr)
    return ts, res.stderr.strip().splitlines()[-1] if res.stderr else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--account", default="<ACCOUNT>")
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--mem", default="120G")
    ap.add_argument("--cpus", type=int, default=12)
    ap.add_argument("--util", type=float, default=0.85,
                    help="gpu_memory_utilization assumed for fit check")
    ap.add_argument("--prefer-tp", type=int, default=None,
                    help="If set and the chosen winner has lower TP, "
                         "tie-break in favour of higher TP within the same "
                         "predicted start window (sec) below.")
    ap.add_argument("--tie-break-window", type=int, default=300)
    ap.add_argument("--prefer-fewer-gpus", action="store_true", default=True,
                    help="Within tie-break window, prefer fewer GPUs (default: "
                         "true). Lower-GPU requests backfill faster on busy "
                         "clusters than --test-only predictions suggest. "
                         "Overridden by --prefer-tp.")
    ap.add_argument("--no-prefer-fewer-gpus", action="store_false",
                    dest="prefer_fewer_gpus")
    args = ap.parse_args()

    print(f"# pick_slot: model={args.model}", file=sys.stderr)
    weight_gb = model_weight_gb(args.model)
    print(f"# weights: {weight_gb:.1f} GB bf16", file=sys.stderr)

    candidates = [c for c in CANDIDATES
                  if fits(weight_gb, c[2], c[3], args.util)]
    if not candidates:
        print(f"ERROR: no candidate fits {weight_gb:.1f} GB at util {args.util}",
              file=sys.stderr)
        return 1

    print("# probing candidates:", file=sys.stderr)
    results = []
    for partition, gpus, tp, per_gpu in candidates:
        ts, msg = probe(partition, gpus, args.account, args.time, args.mem,
                         args.cpus)
        results.append((ts, partition, gpus, tp, per_gpu, msg))
        ts_str = ts.isoformat() if ts else "unschedulable"
        print(f"#   {partition:30s} gpus={gpus} tp={tp} -> {ts_str}",
              file=sys.stderr)

    runnable = [r for r in results if r[0] is not None]
    if not runnable:
        print("ERROR: every candidate is unschedulable", file=sys.stderr)
        return 1

    runnable.sort(key=lambda r: r[0])
    best = runnable[0]

    if args.prefer_tp is not None:
        window = dt.timedelta(seconds=args.tie_break_window)
        within = [r for r in runnable if r[0] - best[0] <= window]
        within.sort(key=lambda r: (-r[3], r[0]))
        best = within[0]
    elif args.prefer_fewer_gpus:
        window = dt.timedelta(seconds=args.tie_break_window)
        within = [r for r in runnable if r[0] - best[0] <= window]
        within.sort(key=lambda r: (r[2], r[0]))
        best = within[0]

    _, partition, gpus, tp, per_gpu, _ = best
    print(f"# winner: {partition} gpus={gpus} tp={tp} per_gpu_gb={per_gpu}",
          file=sys.stderr)

    print(f"PARTITION={partition}")
    print(f"GPUS={gpus}")
    print(f"TP={tp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
