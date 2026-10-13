"""Collect resumable, paired FIA full/no-combine msprof measurements.

Requires cases.json, builds.json, schedules-full.json, schedules-no-combine.json
and validation-full.json in --artifacts. Use one physical card for all rounds.
Reject entire captures on foreign processes during the measurement window or a
>3x median duration tail. Profiler export runs after the measurement window.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def processes(device, path):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True)
    path.write_text(result.stdout + result.stderr)
    active = {int(pid) for card, pid in re.findall(
        r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|', result.stdout, re.MULTILINE) if int(card) == device}
    if not active and f"No running processes found in NPU {device} " not in result.stdout:
        raise RuntimeError("Cannot establish device occupancy")
    return active


def descendants(pid):
    owned = {pid}
    for _ in range(8):
        new = set()
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(") ", 1)[1].split()
                if int(fields[1]) in owned:
                    new.add(int(path.parent.name))
            except (OSError, ValueError, IndexError):
                pass
        if new <= owned:
            break
        owned |= new
    return owned


def stop(proc):
    # Each experiment has its own process group; no other user's job is signalled.
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def parse_capture(directory, reference, plans):
    payload = json.loads((directory / "outputs.json").read_text())
    meta = payload["metadata"]
    for name, digest in meta["input_sha256"].items():
        assert digest == reference["input_sha256"][name]
    summaries = list((directory / "prof").rglob("op_summary*.csv"))
    assert len(summaries) == 1, summaries
    with summaries[0].open() as file:
        rows = list(csv.DictReader(file))
    rows = sorted((r for r in rows if "FusedInferAttentionScore" in r["Op Name"]),
                  key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
    stride = meta["warmup"] + meta["measured"]
    assert len(rows) == stride * len(meta["groups"]), (len(rows), len(meta["groups"]))
    groups, samples = [], []
    for idx, group in enumerate(meta["groups"]):
        selected = rows[idx * stride + meta["warmup"]:(idx + 1) * stride]
        plan = plans[group["case"], group["return_lse"]]
        assert {int(r["Block Num"]) for r in selected} == {plan["aic"]}
        assert {int(r["Mix Block Num"]) for r in selected} == {plan["aiv"]}
        times = [float(r["Task Duration(us)"]) for r in selected]
        groups.append(dict(group, mean_us=statistics.mean(times), median_us=statistics.median(times),
                           min_us=min(times), max_us=max(times), std_us=statistics.pstdev(times), samples=len(times)))
        samples.extend(dict(group, sample=i, duration_us=t) for i, t in enumerate(times))
    return dict(metadata=meta, groups=groups, accuracy=payload["accuracy"],
                op_summary=str(summaries[0]), directory=str(directory)), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--idle-timeout", type=int, default=180)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    configuration = {key: getattr(args, key) for key in
                     ("device", "chunk_size", "warmup", "iterations", "rounds")}
    config_file = root / "collection-config.json"
    if config_file.exists():
        assert json.loads(config_file.read_text()) == configuration, "Resume with the original collection parameters"
    else:
        save(config_file, configuration)
    cases = json.loads((root / "cases.json").read_text())
    builds = {b["variant"]: b for b in json.loads((root / "builds.json").read_text())}
    reference = json.loads((root / "validation-full.json").read_text())["metadata"]
    plans = {}
    for variant in ("full", "no-combine"):
        records = json.loads((root / f"schedules-{variant}.json").read_text())["records"]
        assert len(records) == len(cases) * 2
        for record in records:
            assert record["is_fd"] and record["binary"]
            binary = Path(record["binary"])
            assert binary.is_relative_to(Path(builds[variant]["vendor"]))
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == record["sha256"]
            key = record["case"], record["return_lse"]
            if variant == "full":
                plans[key] = record
            else:
                for field in ("arrays", "aic", "aiv", "tiling_key", "metrics"):
                    assert record[field] == plans[key][field], (key, field)
                assert record["sha256"] != plans[key]["sha256"]
    runs = json.loads((root / "runs.json").read_text()) if (root / "runs.json").exists() else []
    assert all(r["device"] == args.device for r in runs), "Do not mix physical cards"
    samples = []
    if runs:
        with (root / "kernel-durations.csv").open() as file:
            samples = list(csv.DictReader(file))
    for chunk, start in enumerate(range(0, len(cases), args.chunk_size)):
        chunk_file = root / f"chunk-{chunk:02d}-cases.json"
        subset = cases[start:start + args.chunk_size]
        if chunk_file.exists():
            assert json.loads(chunk_file.read_text()) == subset
        save(chunk_file, subset)
        for rid in range(1, args.rounds + 1):
            order = ("full", "no-combine") if rid % 2 else ("no-combine", "full")
            for variant in order:
                if any((r["chunk"], r["round"], r["variant"]) == (chunk, rid, variant) for r in runs):
                    continue
                for attempt in range(1, 21):
                    directory = root / f"chunk{chunk:02d}-r{rid}-{variant}-a{attempt}"
                    if directory.exists():
                        continue
                    directory.mkdir()
                    started = time.monotonic()
                    while processes(args.device, directory / "device-before.log"):
                        if time.monotonic() - started > args.idle_timeout:
                            raise RuntimeError("Device remained busy; accepted captures saved, rerun to resume")
                        time.sleep(3)
                    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.device),
                               ASCEND_CUSTOM_OPP_PATH=builds[variant]["vendor"],
                               ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
                    for key in ("LD_PRELOAD", "ASCEND_PROCESS_LOG_PATH", "ASCEND_LAUNCH_BLOCKING"):
                        env.pop(key, None)
                    cmd = ["msprof", "--ai-core=off", "--task-time=on", "--ascendcl=off",
                           "--output=" + str(directory / "prof"), sys.executable, "-u",
                           str(Path(__file__).with_name("profile_fia_fd_combine_batches.py")),
                           "--cases", str(chunk_file), "--variant", variant, "--mode", "profile",
                           "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                           "--check-device-occupancy",
                           "--output", str(directory / "outputs.pt")]
                    print("Profiling", directory.name, flush=True)
                    reason = None
                    with (directory / "run.log").open("w") as file:
                        proc = subprocess.Popen(cmd, env=env, cwd="/tmp", stdout=file,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                        known = {proc.pid}
                        try:
                            monitor = 0
                            while proc.poll() is None:
                                time.sleep(2)
                                known |= descendants(proc.pid)
                                active = processes(args.device, directory / f"device-during-{monitor}.log")
                                known |= descendants(proc.pid)
                                monitor += 1
                                progress = (directory / "run.log").read_text(errors="replace")
                                measuring = '{"case_begin":' in progress and 'Completed ' not in progress
                                if active - known and measuring:
                                    reason = "Foreign NPU processes observed; entire capture excluded"
                                    if proc.poll() is None:
                                        stop(proc)
                                    break
                        finally:
                            if proc.poll() is None:
                                stop(proc)
                    processes(args.device, directory / "device-after-export.log")
                    if "FOREIGN_NPU_PROCESS:" in (directory / "run.log").read_text(errors="replace"):
                        reason = "Foreign process in pre/post measurement snapshot; entire capture excluded"
                    if reason:
                        (directory / "excluded.txt").write_text(reason + "\n")
                        print(reason, flush=True)
                        continue
                    if proc.returncode:
                        raise RuntimeError(f"Experiment failed: {directory / 'run.log'}")
                    record, durations = parse_capture(directory, reference, plans)
                    if any(g["max_us"] > 3 * g["median_us"] for g in record["groups"]):
                        (directory / "excluded.txt").write_text("Duration exceeds 3x median; entire capture excluded\n")
                        save(directory / "tail-quality.json", record["groups"])
                        continue
                    record.update(chunk=chunk, round=rid, variant=variant, device=args.device, command=cmd)
                    runs.append(record)
                    samples.extend(dict(chunk=chunk, round=rid, variant=variant, **s) for s in durations)
                    save(root / "runs.json", runs)
                    with (root / "kernel-durations.csv").open("w") as file:
                        writer = csv.DictWriter(file, fieldnames=("chunk", "round", "variant", "case",
                                                                "return_lse", "sample", "duration_us"))
                        writer.writeheader()
                        writer.writerows(samples)
                    print(f"Accepted {len(runs)} captures / {len(samples)} durations", flush=True)
                    break
                else:
                    raise RuntimeError("Retry limit reached; accepted captures preserved")
    print("All paired rounds complete", flush=True)


if __name__ == "__main__":
    main()
