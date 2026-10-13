"""Compare complete FA/FIA FD kernel durations on the same idle physical NPU.

Requires builds.json (current full FA), cases-fa0.json, cases-fa16.json,
cases.json (FIA), validation-{fa0,fa16,fia}.json and schedules-fia.json.
Runs LSE on/off and rotates FA auto, FA split16, FIA auto in three rounds.
No combine-disabled binary is used. Raw and rejected captures are retained.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
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
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


from collect_fd_combine_ab import parse_capture as parse_fa_capture
from collect_fia_fd_combine_sweep import parse_capture as parse_fia_capture


def parse_capture(directory, variant, references, plans):
    if variant == "fia":
        return parse_fia_capture(directory, references[variant], plans)
    return parse_fa_capture(directory, references[variant])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--variants", nargs="+", default=["fa0", "fa16", "fia"], choices=("fa0", "fa16", "fia"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--idle-timeout", type=int, default=180)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    config = {k: getattr(args, k) for k in ("device", "variants", "rounds", "warmup", "iterations")}
    if (root / "collection-config.json").exists():
        assert json.loads((root / "collection-config.json").read_text()) == config
    save(root / "collection-config.json", config)
    builds = json.loads((root / "builds.json").read_text())
    for variant in args.variants:
        if variant == "fia":
            continue
        record = builds[variant]
        assert hashlib.sha256(Path(record["native"]).read_bytes()).hexdigest() == record["sha256"]
    references = {v: json.loads((root / f"validation-{v}.json").read_text())["metadata"]
                  for v in args.variants}
    records = json.loads((root / "schedules-fia.json").read_text())["records"]
    plans = {(r["case"], r["return_lse"]): r for r in records}
    case_names = {c["name"] for c in json.loads((root / "cases.json").read_text())}
    for name in case_names:
        for lse in (True, False):
            record = plans[name, lse]
            assert "is_fd" in record  # Non-FD FIA routes are reported separately.
            assert hashlib.sha256(Path(record["binary"]).read_bytes()).hexdigest() == record["sha256"]
    runs = json.loads((root / "runs.json").read_text()) if (root / "runs.json").exists() else []
    assert all(r["device"] == args.device for r in runs), "Do not mix physical cards"
    samples = []
    if runs:
        with (root / "kernel-durations.csv").open() as file:
            samples = list(csv.DictReader(file))
    for rid in range(1, args.rounds + 1):
        offset = (rid - 1) % len(args.variants)
        order = args.variants[offset:] + args.variants[:offset]
        for variant in order:
            if any((r["round"], r["variant"]) == (rid, variant) for r in runs):
                continue
            for attempt in range(1, 21):
                directory = root / f"r{rid}-{variant}-a{attempt}"
                if directory.exists():
                    continue
                directory.mkdir()
                started = time.monotonic()
                while processes(args.device, directory / "device-before.log"):
                    if time.monotonic() - started > args.idle_timeout:
                        raise RuntimeError("Device remained busy; accepted captures saved for resume")
                    time.sleep(3)
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(args.device),
                           PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                           ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
                for key in ("LD_PRELOAD", "FA_FD_TILING_CAPTURE", "ASCEND_CUSTOM_OPP_PATH",
                            "ASCEND_PROCESS_LOG_PATH", "ASCEND_LAUNCH_BLOCKING"):
                    env.pop(key, None)
                if variant == "fia":
                    runner = "profile_fia_fd_combine_batches.py"
                    selector = ["--cases", str(root / "cases.json"), "--variant", "full"]
                else:
                    runner = "profile_fd_combine_sweep.py"
                    selector = ["--cases-json", str(root / f"cases-{variant}.json"),
                                "--manifest", str(root / "builds.json"), "--variant", variant]
                cmd = ["msprof", "--ai-core=off", "--task-time=on", "--ascendcl=off",
                       "--output=" + str(directory / "prof"), sys.executable, "-u",
                       str(Path(__file__).with_name(runner)), *selector,
                       "--mode", "profile", "--warmup", str(args.warmup),
                       "--iterations", str(args.iterations), "--check-device-occupancy",
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
                                reason = "Foreign NPU process during capture; entire capture excluded"
                                if proc.poll() is None:
                                    stop(proc)
                                break
                    finally:
                        if proc.poll() is None:
                            stop(proc)
                processes(args.device, directory / "device-after-export.log")
                if "FOREIGN_NPU_PROCESS:" in (directory / "run.log").read_text(errors="replace"):
                    reason = "Foreign NPU process before/after benchmark calls; entire capture excluded"
                if reason:
                    (directory / "excluded.txt").write_text(reason + "\n")
                    print(reason, flush=True)
                    continue
                if proc.returncode:
                    raise RuntimeError(f"Experiment failed: {directory / 'run.log'}")
                record, durations = parse_capture(directory, variant, references, plans)
                if any(g["max_us"] > 3 * g["median_us"] for g in record["groups"]):
                    (directory / "excluded.txt").write_text("Duration exceeds 3x median; entire capture excluded\n")
                    save(directory / "tail-quality.json", record["groups"])
                    continue
                record.update(round=rid, variant=variant, device=args.device, command=cmd)
                runs.append(record)
                samples.extend(dict(round=rid, variant=variant, **s) for s in durations)
                save(root / "runs.json", runs)
                with (root / "kernel-durations.csv").open("w") as file:
                    writer = csv.DictWriter(file, fieldnames=("round", "variant", "case", "return_lse",
                                                            "sample", "duration_us"))
                    writer.writeheader()
                    writer.writerows(samples)
                print(f"Accepted {len(runs)} captures / {len(samples)} durations", flush=True)
                break
            else:
                raise RuntimeError("Retry limit reached; accepted captures preserved")
    print("All paired rounds complete", flush=True)


if __name__ == "__main__":
    main()
