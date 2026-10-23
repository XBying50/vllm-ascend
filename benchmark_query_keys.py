"""Profile actual query-key functions; compare a frozen source to this checkout."""

import argparse
import ast
import cProfile
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import time
import tracemalloc

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler


def load_baseline(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVPoolScheduler")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_generate_store_query_keys")
    namespace = vars(pool_scheduler).copy()
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[fn.name]


def measure(fn, obj, hashes, include_layers, loops):
    start = time.perf_counter_ns()
    for _ in range(loops):
        fn(obj, hashes, include_layers, 1)
    return (time.perf_counter_ns() - start) / loops / 1e3


def profile(fn, obj, hashes, include_layers):
    profiler = cProfile.Profile()
    profiler.runcall(fn, obj, hashes, include_layers, 1)
    stats = pstats.Stats(profiler)
    calls = [
        {"file": file, "line": line, "function": name, "calls": values[1], "self_s": values[2], "total_s": values[3]}
        for (file, line, name), values in stats.stats.items()
    ]
    gc.collect()
    tracemalloc.start()
    keys = fn(obj, hashes, include_layers, 1)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"calls": calls, "retained_bytes": current, "peak_bytes": peak, "key_count": sum(map(len, keys))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    baseline = load_baseline(args.baseline)
    candidate = pool_scheduler.KVPoolScheduler._generate_store_query_keys
    functions = {"baseline": baseline, "candidate": candidate}
    # blocks, TP, put_step, DCP, PP, PCP, layers, include_layers
    cases = {
        "empty": (0, 8, 1, 2, 4, 2, 32, False),
        "single": (1, 1, 1, 1, 1, 1, 1, False),
        "single_layerwise": (1, 2, 1, 1, 2, 1, 4, True),
        "long_prompt": (1024, 1, 1, 1, 1, 1, 1, False),
        "multi_pp_heads": (256, 8, 1, 1, 4, 1, 32, False),
        "multi_cp_pp_heads": (128, 8, 1, 4, 4, 2, 32, False),
        "replicated_kv_heads": (512, 8, 4, 2, 2, 2, 16, False),
        "layerwise": (128, 8, 1, 2, 4, 2, 32, True),
    }
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "dependency_mode": "CPU; repository _mock_deps; actual key classes and scheduler method",
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(pool_scheduler.__file__).read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "cases": {},
    }
    for name, config in cases.items():
        blocks, tp, put_step, dcp, pp, pcp, layers, include_layers = config
        obj = pool_scheduler.KVPoolScheduler.__new__(pool_scheduler.KVPoolScheduler)
        for attr, value in {
            "tp_size": tp, "put_step": put_step, "dcp_size": dcp, "pp_size": pp,
            "pcp_size": pcp, "pp_rank": pp - 1, "pp_layer_offset": layers,
            "num_layers": layers, "model_name": "Qwen3-32B", "kv_cache_group_families": ["default", "c4"],
        }.items():
            setattr(obj, attr, value)
        hashes = [i.to_bytes(32, "big") for i in range(blocks)]
        expected = baseline(obj, hashes, include_layers, 1)
        assert candidate(obj, hashes, include_layers, 1) == expected, name
        del expected
        sample_us = measure(baseline, obj, hashes, include_layers, 3)
        loops = max(1, min(10000, int(120000 / max(sample_us, 0.1))))
        timings = {label: [] for label in functions}
        for repeat in range(args.repeats):
            for label in (list(functions) if repeat % 2 == 0 else list(functions)[::-1]):
                gc.collect()
                timings[label].append(measure(functions[label], obj, hashes, include_layers, loops))
        row = {"config": config, "loops_per_repeat": loops, "latency_us": timings, "profiles": {}}
        for label, fn in functions.items():
            row["profiles"][label] = profile(fn, obj, hashes, include_layers)
        row["speedup"] = statistics.median(timings["baseline"]) / statistics.median(timings["candidate"])
        report["cases"][name] = row
        print(name, "baseline_us", round(statistics.median(timings["baseline"]), 3),
              "candidate_us", round(statistics.median(timings["candidate"]), 3),
              "speedup", round(row["speedup"], 3), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
