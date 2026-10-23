# NPU validation status — 2026-10-23

These checks used one physical Ascend 910B3 (64 GB), CANN 9.0.0 and driver 25.5. They do not establish end-to-end acceptance.

| Check | Observed result | Conclusion |
| --- | --- | --- |
| Real NPU matrix multiply | Exact FP32 reference match, process exit 0 | Arithmetic preflight passed |
| Memcache 1.2.0/1.2.1 with NPU buffers | Four old-to-new/new-to-old normal/layerwise cases passed data equality and hit/miss assertions; 192 keys; process abort 134 during teardown | Overall failed; assertions alone are not a passing integration run |
| Memcache 1.2.1 init/close only | Abort 134 without importing either query-key implementation | Teardown failure can occur without the proposed change |
| Mooncake 0.3.11.post1 import only | SIGABRT at exit; no query-key code executed | Independent runtime failure reproduced; gdb stack enters CANN libascend_trace.so finalization |
| Qwen3-0.6B inference preflight | Loaded 1.14 GiB; failed before generation because vllm_ascend.vllm_ascend_C is missing | No model output or throughput result |
| Native extension build | Fails in quant_lightning_indexer_v2_infershape.cpp:65: OP_LOGE_FOR_INVALID_VALUE is unavailable in the installed CANN SDK | Compatible native build remains required |

The Memcache tests use real storage/NPU buffers and synthetic scheduler topology inputs on one card. They are not actual multi-card TP/PP deployments. The minimal teardown reproductions establish that the candidate is unnecessary to trigger failure; they do not identify the original memory corruption site or prove all backend issues share one cause.

## Versions and reproducibility

- vLLM-Ascend base: 3c088bc78b97e0fb9e5dc380a93a94c23fd77dc0.
- vLLM: c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c.
- Python 3.12.9; Torch 2.10.0+cpu; torch-npu 2.10.0.post4.
- Isolated .venv-npu overlay; the validated .venv-full CPU environment is preserved.
- Memcache/Memfabric 1.2.1; Mooncake-transfer-engine-npu 0.3.11.post1.
- Official pinned FLA A2 wheel 26.9.1+dev5cc6990, SHA-256 2577eca3f1015ca83897bf638046f41cb54691a0c15c433d9e43946fa32d985b.
- Qwen/Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca; no remote code execution.
- The full upstream Docker dependency stack was not reproduced: its pinned Memcache/Memfabric 1.3.0 download returned 404, and the local Triton version is 3.2.1 rather than required 3.2.2.

Local scripts are tools/npu_smoke.py, tools/npu_model_smoke.py and tools/memcache_key_integration.py. The last accepts --init-only for the minimal store lifecycle reproduction. Hardware jobs take the shared runtime lock, use process-group timeouts and clean up their own metadata services. No shared SDK, huge-page setting or device reset was changed.

Raw evidence: out/npu-smoke.json, out/memcache-key-integration-1.2.1-result.json, out/memcache-init-only-result.json, out/memcache-init-only.log, out/npu-import-isolation.json, out/mooncake-exit-backtrace.log, out/npu-model-smoke-with-fla.log and out/npu-native-build.log.

Remaining: a matching native/runtime stack, model output and cache-hit comparisons in both directions, end-to-end latency/throughput, and actual multi-card validation on suitable hardware. See NPU_VALIDATION_PLAN.md.
