# AscendStore query-key optimization: review evidence

Candidate: [5b74b11745ac7ba703539e2d23d0019c30e8430b](https://github.com/XBying50/vllm-ascend/commit/5b74b11745ac7ba703539e2d23d0019c30e8430b). Base: 3c088bc78b97e0fb9e5dc380a93a94c23fd77dc0. Related upstream issue: [14146](https://github.com/vllm-project/vllm-ascend/issues/14146). This evidence branch is separate from the proposed source diff; no upstream PR has been submitted.

The first block uses canonical PoolKey/LayerPoolKey serialization. From the second block, the method reuses prefixes derived from those strings and appends the new hash. Prefixes live only within one call. Namespace, rank/layer ordering and the singleton path are covered by compatibility tests.

## CPU evidence

Official AscendStore suite: **939 passed, 12 existing skips, 0 failures** in 65.62 s. Required `bash format.sh ci` checks passed. The new construction-count regression fails on the original implementation; five new compatibility tests also pass on that original implementation.

Measured on an ARM64 host, Python 3.12.9, Torch 2.10.0+cpu, vLLM c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c. The real scheduler method and serializers are measured with repository CPU mocks for device/backend dependencies. Topologies below are method inputs, not launched multi-card deployments. Values are medians of five paired rounds in alternating order.

| Case | Keys | Original µs | Candidate µs | Speedup |
| --- | ---: | ---: | ---: | ---: |
| empty | 0 | 0.475 | 0.490 | 0.97x |
| single | 1 | 5.695 | 5.766 | 0.99x |
| single_layerwise | 8 | 27.211 | 27.135 | 1.00x |
| long_prompt | 1024 | 5206.284 | 757.020 | 6.88x |
| multi_pp_heads | 8192 | 30378.638 | 1163.069 | 26.12x |
| multi_cp_pp_heads | 16384 | 63241.287 | 3225.189 | 19.61x |
| replicated_kv_heads | 4096 | 16165.907 | 777.093 | 20.80x |
| layerwise | 65536 | 153445.355 | 14387.695 | 10.67x |

All samples, exact workload tuples, cProfile calls, memory values and baseline/candidate file hashes are in [results/benchmark.json](results/benchmark.json). Extra traced peak memory is at most 68,600 bytes (0.53%) for the measured cases; this is not a universal bound. Per-call extra space is O(rank/layer combinations × prefix length). The return-list size remains O(blocks × rank/layer combinations). No cross-request cache is retained.

## Reproduce

Use the official repository CPU-unit-test dependency environment at the versions above. Clone the source branch and this evidence branch separately, then run from the evidence checkout:

```bash
python run_cpu_benchmark.py /path/to/candidate-checkout \
  --baseline "$PWD/baseline-pool_scheduler.py" \
  --output "$PWD/reproduced-benchmark.json" --repeats 5
```

The wrapper prepends its own `cpu-only-bin/npu-smi` stub to select the repository's CPU-test branch for this process. It does not access the NPU. Never use that wrapper or stub for hardware validation. The wrapper is a packaging convenience; the recorded five-round result predates it and measures the same unchanged method and benchmark.

To run CPU unit tests from the candidate checkout with the same environment (set `EVIDENCE` to this evidence checkout):

```bash
PATH="$EVIDENCE/cpu-only-bin:$PATH" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python -m pytest -q tests/ut/distributed/ascend_store
bash format.sh ci
```

The baseline source is included for AST loading of the exact original method; the rest of the runtime comes from the candidate checkout. The performance regression is structural, counting metadata construction, so it does not rely on unstable wall-clock thresholds.

## Hardware status

[NPU_VALIDATION_STATUS.md](NPU_VALIDATION_STATUS.md) records attempts and failures. Real arithmetic passed. Four Memcache/NPU-buffer compatibility cases passed their data/hit assertions, but the processes aborted during cleanup; initialization-only also aborts without candidate imports, so overall backend integration remains failed. Model inference is blocked by a missing native extension whose build fails against the local CANN API. No end-to-end model throughput, model accuracy or distributed-hardware success is claimed.

This change and its supporting materials were developed with OpenAI Codex assistance. Human review, community registration approval, compatible runtime validation and the upstream release fragment remain prerequisites for the relevant publication/acceptance stages.
