# Findings: flash-attention decode (head256, nb=1) on Maxwell (Tesla M40, cc 5.2)

Target shape: `FLASH_ATTN_EXT(hsk=256,hsv=256,nh=12,nr23=[1,1],nb=1,mask=1,type f16)` - Qwen3.6-27B decode.
Baseline op-level: **257 GFLOPS @ kv=16384, 261 @ kv=65536** (~37 GB/s effective vs ~264 GB/s ceiling).
Dispatch: `ggml_cuda_get_best_fattn_kernel()` (fattn.cu ~358) sends this to the **VEC kernel** (fattn-vec.cuh), not the tile kernel. VEC streams K/V direct from global fp16, no staging.

Three independent attempts to beat the baseline all FAILED and were fully reverted (git diff = 0).

## Attempt 1 - knob tuning (prior session, HANDOFF.md)
TILE ncols=1 (228/245), TILE nbatch_K=128 (229/245), VEC nthreads 64/256 (254), skip-VKQ-rescale (255). All < baseline.

## Attempt 2 - split-K via existing machinery + wider loads
- The VEC path already has split-K: `launch_fattn()` (fattn-common.cuh:1151) picks `parallel_blocks`, `flash_attn_combine_results()` merges partials online-softmax.
- (f) Maxwell-gated occupancy search raising `parallel_blocks`: **FLAT 257/261** for pb=1..32 (48 blocks / 24 SMs at kv=16384). GOTCHA: the pre-existing efficiency loop (fattn-common.cuh:1183) starts `efficiency_percent_best=0` and silently OVERRIDES any `parallel_blocks` chosen before it (locks pb=1 at 50% eff) - any search must run AFTER it.
- (g) lift `ggml_cuda_get_max_cpy_bytes()` 8->16 on cc>=500 (common.cuh): **FLAT 256/262**.
- (h) `__byte_perm`+`HADD2` half2->float2 unpack in `ggml_cuda_mad` float2 path: **FLAT 257/261**.

Conclusion: the 37 GB/s wall is the serialized max/exp/softmax + KQ shared round-trip chain per 128-row block, NOT K-load MLP or dot-product ALU.

## Attempt 3 - purpose-built decode kernel (this session)
New `fattn-vec-decode.cuh`: 8 warps, warp-per-K-row dot phase writing exp scores to a shared KQ tile, then a column-V-accumulation phase (threads split D, read V columns stride-nb21). Per-warp VKQ shared slabs (no cross-warp register shuffles). Plus a Maxwell-gated dispatch hook in `ggml_cuda_flash_attn_ext()` routing D=256 fp16-K/V nb=1 to it.

- OP-LEVEL: **25-94 GFLOPS** - 3-10x SLOWER than baseline. The strided V reads (column accumulation = gather, ~64 transactions/row) and low per-thread MLP dominate; conflict-free per-warp VKQ slabs forced 106 registers -> poor occupancy. Could not reach VEC perf after several loop-restructure iterations.
- CORRECTNESS: unresolved sentinel-overflow (12 test FAILs, all sinks=0 nb=1 hsk=256). Debug findings: pb=1 main kernel isolated PASSED; compute-sanitizer memcheck = 0 OOB; SASS STG stores present; dst write indexing verified == VEC (`[D, ne02, ne01, ne03]`, head inner); scale/ALiBi/mask verified; not graph-related (GGML_CUDA_DISABLE_GRAPHS same); device printf confirmed real values land in dst, yet the graph-harness compare reads a corrupted/mismatched buffer. Root cause NOT found.
- E2E llama-bench (before correctness fix): baseline tg16=11.01, d16384=8.35 - NO win.

LESSON: replicating `launch_fattn`'s buffer/workspace contract (the f16 extra workspace from `ggml_cuda_flash_attn_ext_get_alloc_size` + the `dst_tmp`/`dst_tmp_meta` pool allocs) outside `launch_fattn` is fragile - the graph/pool harness mishandles a hand-rolled launcher. Any future decode kernel must reuse `launch_fattn`'s alloc/dispatch plumbing rather than a separate launcher.

## Bottom line
The ~37 GB/s VEC wall appears to be a hard structural floor for this fp16 head256 decode shape on Maxwell. Knob tuning, split-K, and a custom kernel all failed to beat 257 GFLOPS. The VEC kernel's design (row-contiguous coalesced K/V loads, register VKQ with per-128-row rescale) is already well-suited; the loss is in the online-softmax serialization which no restructure attempted here could remove without a much deeper redesign.

Files: all changes reverted. This file + HANDOFF.md document the negative results.
