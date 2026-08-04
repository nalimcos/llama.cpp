# Handoff: llama.cpp Tesla M40 (cc 5.2) performance work - next investigation session

## TOP PRIORITY: MTP models crash (new, blocking)

The user added MTP variants of the 27B and BOTH crash on this build:
- `~/gguf/Qwen3.6-27B-MTP-Q4_0.gguf` (16.05 GB) - crashes BEFORE prompt processing.
- `~/gguf/Qwen3.6-27B-MTP-Q4_K_S.gguf` (16.12 GB) - crashes AFTER prompt processing, on the first draft/decode step.

Crash signature (from user log, Q4_K_S via llama-server, `--spec-type draft-mtp`, `-fa on -ngl all`, `-ctk/-ctv q8_0`, `-b 1024 -ub 1024`, `-c 100000`):
- pp completes: `prompt processing, n_tokens = 828, 63.03 tokens per second` (note: ~63 t/s matches the dense 27B pp).
- Then: `ggml-cuda.cu:106: CUDA error: an illegal memory access was encountered`
  in `ggml_backend_cuda_synchronize` (ggml-cuda.cu:2589, `cudaStreamSynchronize`).
- Stack: `common_speculative_impl_draft_mtp::draft` -> `common_sampler_sample` -> `llama_context::synchronize` -> `ggml_backend_sched_synchronize`.
- So the illegal access is enqueued during the MTP DRAFT model's decode graph, surfaced at the next sync.

## Session findings - MTP crash root-caused, two SEPARATE bugs

### Bug A: Q4_K_S post-pp draft crash = CUDA graphs on Maxwell (exposed by our commit 26827b1d9)

- compute-sanitizer: invalid 4-byte reads in `k_get_rows_float<float,float>` (getrows.cu:73), BOTH via `cudaGraphLaunch` (4x) and direct `cudaLaunchKernel` (3x), all ~3.8-3.9 GB past a small 8MB buffer.
- `GGML_CUDA_DISABLE_GRAPHS=1` does NOT help (env var read once via a `static` in `ggml_cuda_graph::is_enabled()`, common.cuh:1258) - still crashes because the OOB also occurs via the graph-update direct re-evaluation path, not only the captured launch.
- KEY BISECT: base `796e62235` (pre-Volta graphs VETOED => disabled on Maxwell) does NOT crash with the identical server config (Q4_K_S, c=100000, draft-mtp); completes 32 tokens, acceptance 0.44. So enabling CUDA graphs on Maxwell (`26827b1d9`) exposes the Q4_K_S draft crash. The draft graph (qwen35 graph_mtp, src/models/qwen35.cpp:489) does GET_ROWS on `cur` w/ `inp_out_ids` (line 634) and on tok_embd (line 524); with graphs on Maxwell the captured/updated kernel reads OOB.
- llama-speculative (no server, same model, c=8192 AND c=100000) does NOT crash - needs the server's slot/KV management + graphs to trigger.
- FIX DIRECTION: simplest safe = disable CUDA graphs for MTP draft contexts on cc 5.x (ctx_type == LLAMA_CONTEXT_TYPE_MTP). The graph-compat check `ggml_cuda_graph_check_compability` only guards MUL_MAT_ID; GET_ROWS is not gated. Alternatively root-cause why the draft graph capture reads stale pointers on Maxwell.

### Bug B: Q4_0 pre-pp crash = PRE-EXISTING upstream bug (NOT from our 9 commits)

- Crashes at `common_context_can_seq_rm` probe decode (common/common.cpp:1508, evals 2 tokens) right after `creating MTP draft context`, surfaced in `llama_kv_cache::clear` -> `ggml_backend_cuda_buffer_clear` (ggml-cuda.cu:910).
- compute-sanitizer: 1 invalid read in `k_get_rows_float<float,float>`, 481 MB past a 347 MB buffer, direct `cudaLaunchKernel` (NOT a graph). The MTP draft ctx probe decode runs with token input but the hidden-state input `inp->h` ("mtp_h_input", qwen35.cpp:530) uninitialized/garbage -> nextn embed GET_ROWS reads a garbage index.
- CONFIRMED pre-existing: base `796e62235` build crashes the SAME way at load with Q4_0. Our commits are innocent.
- CPU (`-dev none`) Q4_0 MTP server works fine (16 tokens) - CUDA-only manifestation (garbage index reads valid-but-wrong or clamped memory on CPU).
- Q4_0 MTP GGUF has NO `blk.64.nextn.embed_tokens.weight` and NO `blk.64.nextn.shared_head_head.weight` (llama-gguf) -> relies on fallback to `model.tok_embd` / `model.output` (qwen35.cpp:522,637). Q4_K_S HAS them. This config difference routes Q4_0 through the probe-decode OOB path; Q4_K_S avoids it.
- Likely a genuine upstream MTP bug: server's `common_context_can_seq_rm` feeds an embd-less batch to a graph needing `inp->h`. Worth reproducing on a clean checkout and reporting upstream.
- llama-bench (plain model, no MTP ctx) on Q4_0 CUDA is 100% clean (0 sanitizer errors, pp32=17.97, tg8=10.01 t/s). The model itself is fine; only the MTP draft context triggers it.

### Bottom line for next session
- Bug A (graph/Maxwell, ours): decide gate-off for MTP draft ctx on cc 5.x vs root-cause the draft graph capture. The 9 commits' perf wins are otherwise solid (test-backend-ops passed, dense+MoE llama-bench fine).
- Bug B (Q4_0 probe OOB): pre-existing upstream; reproduce on clean checkout and report. Blocks the Q4_0 MTP model but not our perf work.

## Environment
- GPU0: Tesla M40 24GB, cc 5.2 (Maxwell), 24 SMs, legacy driver <=580, CUDA 12.x. GPU1: GTX 1050 2GB cc 6.1 (correctness spot-check only, `-b CUDA1`).
- Repo: `/home/milan/llama.cpp/llama.cpp`. Build: `cmake --build build --config Release -j22` (add `--target test-backend-ops` for fast op iteration). Binaries in `build/bin`.
- llama-cli is a server-spawning wrapper; use llama-bench and test-backend-ops for timing.
- NEVER run two GPU benchmarks/builds at once. Builds+benchmarks take minutes; only rebuild+rebench when a change genuinely requires it. Check GPU free first: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`; kill stale llama/test processes holding VRAM.
- Redirect long command output to a file and read the file (terminal output is lost after a tool timeout).

## Critical measured context (trust; do not re-derive)
- TRUE device ceiling: ~264 GB/s read-only, ~233 GB/s read+write (float4 grid-stride microbenchmark). Theoretical 288 GB/s. The old "189-200 GB/s" was the GEMV-with-dequant COMPUTE ceiling, NOT device bandwidth - there IS ~28% untapped bandwidth on paper, but it's compute/dequant-bound, not memory-bound.
- Maxwell cc 5.2: NO dp4a (cc>=6.1), NO fp16 math, NO tensor cores/MMA, NO fast_fp16, NO MMQ (needs dp4a), NO MMF. All quantized GEMV/GEMM use FP32/scalar kernel variants; pp-sized quantized GEMM falls back to dequantize-to-FP32 + cuBLAS SGEMM.
- `fast_fp16_available(cc)` false on Maxwell -> FA tile uses `ggml_cuda_fattn_tile_get_config_nvidia_fp32` (fattn-tile.cuh:90) + `#else` (non-FAST_FP16_AVAILABLE) branches. Editing that FP32 table is inherently cc5.x-gated.

## Commits this session (all local on master, 9 ahead of origin, cc5.x-gated, NOT pushed)
Base b10219 `796e62235` (+ prior local `3cbe2fa23` mul_mat_id host-sort - do not revert).
1. `26827b1d9` probe CUDA graph capture support instead of pre-Volta veto (graphs work on Maxwell w/ 580 driver).
2. `8c8dfdd1f` tune fused MMVQ mul_mat_id for Maxwell (moe rpb=8; mmid gates Q4_0/Q4_K/Q8_0 -> 8).
3. `92d4cf950` single warp/row for MMVQ decode ncols_dst=1.
4. `e89db73af` rpb=4 for MMVQ decode ncols_dst=1. MUL_MAT_ID n=1: q4_0 499->628, q4_K 425->536 GFLOPS; dense GEMV q4_0 669->723, q4_K 550->647.
5. `925cdb2a9` Q6_K mmid max batch 4->8: n=8 175->390 GFLOPS.
6. `312f63df6` best-guess mmid max batch 8 for remaining K-quants (Q2_K/Q3_K/Q4_1/Q5_0/Q5_1/Q5_K).
7. `5a233fbcf` 2 warps/row for IQ2/IQ3 decode: iq2_xxs 477->564, iq2_xs 477->521, iq3_s 411->436 GFLOPS (IQ1/IQ4 stay at 1).
8. `32e00785f` FA tile ncols=1 decode path. nb=1 kv=16384: hsk64 224->804, hsk128 247->790 GFLOPS (3.2-3.6x). Helps GQA ratio 1-2 models only.
9. `1a539e344` chunk medium MoE batches (8 < n_tokens <= 128) through fused MMVQ kernel instead of synchronizing fallback; re-enables CUDA graphs for whole pp graph. MUL_MAT_ID Q4_K n=32: 310->627 GFLOPS. New macro `MMVQ_MOE_CHUNK_MAX_BATCH=128` in mmvq.cuh.

## Models (`~/gguf/`) and current perf (M40, -fa on -ngl 99)
- `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (MoE 128e/8a, 40L, head128, GQA8): tg 37.7 -> 40.3 t/s; pp512=253; pp8192(-ub8192)=337. pp at ub=32: pp128/512 85-87 -> 154-157 t/s (+80%); ub=128: ~150->199 t/s (+32%); ub=2048 unchanged.
- `Qwen3.6-27B-Q4_K_S.gguf` (DENSE, 64L, embd5120, head256, GQA6): tg 9.07 -> 10.88 t/s; pp512/8192 ~73 t/s. Bandwidth-bound dense (all 26.9B params/token); near structural floor.
- `Qwen3.6-27B-MTP-Q4_0.gguf` / `Qwen3.6-27B-MTP-Q4_K_S.gguf` - NEW, crash (see TOP PRIORITY).
- `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` (IQ4_XS/IQ3_S mix): tg ~37 t/s, pp512=215. IQ4_XS dense GEMV is fastest (797 GFLOPS); IQ3_S decode improved by IQ warps change.
- `Qwen3.6-35B-Q4_0.gguf` - sloppy requant, loops in generation; timing only. tg ~43 t/s.
- gemma-4-* / Qwen3.5-9B - spot-check / smaller refs.

## Remaining performance research directions (next session)
1. MTP crash fix (TOP - see above). Once fixed, MTP speculative decode should benefit directly from the mmvq n=2-8 tuning (verification batches).
2. 27B decode is Q4_K-GEMV-bound (~143-157 GB/s vs 264 ceiling). The Q4_K vec_dot is limited by byte-packed scale unpacking (Q4_0/Q5_K/Q6_K/IQ4_XS all hit their ceilings; only Q4_K lags ~15%). A float-FFMA vec_dot prototype REGRESSED (613 vs 660 GFLOPS) - the int path with accumulator reuse is near-optimal. Remaining lever: a Maxwell-specific Q4_K dot that reduces the scale-unpack cost (e.g. __byte_perm) without losing accumulator reuse - hard, low expected ROI.
3. FA decode: GQA-packed paths (both target models) are at structural ceiling (head128 ~171 GB/s, head256 ~112 GB/s). head256 is hard-capped by the 64KB shared wall + fp32 KV staging (KV_tmp is 78% of shared; nbatch_fa=64 impossible, 2-blocks/SM perf-neutral). Fixing needs a new KQ-accumulation design - out of scope for minimal change. The ncols=1 win (GQA 1-2 models) is committed.
   - UPDATE (head256 nb=1 nr=[1,1] decode): INVESTIGATED - NEGATIVE RESULT, reverted, do not redo. KEY CORRECTION: the nr=[1,1] head256 decode dispatches to the VEC kernel (kernel=100 in ggml_cuda_get_best_fattn_kernel, fattn.cu), NOT the tile kernel. Baseline FLASH_ATTN_EXT(hsk=256,hsv=256,nh=12,nr23=[1,1],nb=1,mask=1,type_K/V=f16): 257 GFLOPS (kv=16384) / 261 (kv=65536); nr=[4,1]: 752-772. The VEC kernel streams K/V direct from global fp16 (no staging, LDG.E.CI cached) but is latency-bound at only ~37 GB/s effective (D=256 halves per-thread MLP: 4 outstanding K loads + dependent expf + KQ shared round-trip + V rescale per 128-row block). Experiments (all reverted, none beat 257 by >10%): (a) route head256 nb=1 to TILE ncols=1 (nbatch_fa=128, dynamic 67.6KB shared via CUDA_SET_SHARED_MEMORY_LIMIT) = 228/245 GFLOPS (fp32 staging overhead dominates, VEC wins); (b) TILE nbatch_K=128 (halve KV_tmp) = 229/245; (c) VEC nthreads 256 = 254; (d) VEC nthreads 64 = 254; (e) VEC skip-VKQ-rescale-when-max-unchanged = 255. Correct dispatch probe: nr=[1,1] nb=1 uses VEC (cols_per_block=1), nr=[4,1] uses TILE ncols=4. Only remaining idea is a purpose-built split-K/pipelined decode kernel (overlapped K/V loads + no KQ shared round-trip) - a new subsystem, beyond minimal-change scope. e2e baseline (llama-bench Qwen3.6-27B-Q4_1, -fa on -ngl 99 -ctk/-ctv q8_0, -p 0 -n 16 -d 0,16384,65536): tg16=11.04, d16384=8.41, d65536=4.98 t/s.
   - UPDATE2 (attempt 2, split-K + wider loads): NEGATIVE RESULT, all reverted, do not redo. The VEC path ALREADY has split-K machinery: launch_fattn (fattn-common.cuh:1151-1207) picks parallel_blocks and flash_attn_combine_results merges partials online-softmax. (f) Maxwell-gated occupancy search raising parallel_blocks (reuses combine kernel) = FLAT 257/261 for pb=1..32 (48 blocks / 24 SMs at kv=16384) - split-K does NOT help. GOTCHA: the pre-existing efficiency loop (fattn-common.cuh:1183) starts efficiency_percent_best=0 and silently OVERRIDES any parallel_blocks chosen before it (locks in pb=1 at 50% eff) - any search must run AFTER it. (g) lift ggml_cuda_get_max_cpy_bytes 8->16 on cc>=500 (common.cuh) = FLAT 256/262. (h) __byte_perm+HADD2 half2->float2 unpack in ggml_cuda_mad float2 path (common.cuh) = FLAT 257/261. Conclusion: the 37 GB/s wall is the serialized max/exp/softmax + KQ shared round-trip chain per 128-row block, NOT K-load MLP or dot-product ALU. No minimal knob fixes it; needs a purpose-built pipelined decode kernel (out of scope). Verified FLAT results by temporarily re-adding TEMP kv=16384/65536 hsk=256 cases (removed again).
   - UPDATE3 (attempt 3, purpose-built decode kernel fattn-vec-decode.cuh): NEGATIVE RESULT, REVERTED (git diff=0), do not redo. Built a new warp-per-K-row dot + column-V-accumulation decode kernel (8 warps, lane-strided Q/K/V, no KQ shared round-trip, per-warp VKQ slabs) + a Maxwell-gated dispatch hook in ggml_cuda_flash_attn_ext (fattn.cu) routing D=256 fp16-K/V nb=1 to it. OP-LEVEL: 25-94 GFLOPS vs VEC baseline 257 - 3-10x SLOWER. The strided V reads (column accumulation = stride-nb21 gather, 64 transactions/row) and low per-thread MLP dominate; per-warp conflict-free VKQ slabs forced 106 regs -> poor occupancy. Could not reach VEC perf. CORRECTNESS: also hit a sentinel-overflow (12 test FAILs, sinks=0 nb=1 hsk=256) that stayed unresolved despite: pb=1 isolation (passed), memcheck clean (0 OOB), SASS STG present, write indexing verified == VEC, scale/ALiBi verified, graphs disabled. The dst write lands (device printf confirmed real values) yet the graph-harness compare reads a corrupted/mismatched buffer - root cause not found; likely a subtle pool/buffer-layout interaction when bypassing launch_fattn's f16 workspace + dst_tmp/dst_tmp_meta allocs. E2E llama-bench (before fixing correctness) = baseline tg16=11.01, d16384=8.35 (no win). LESSON: replicating launch_fattn's buffer/workspace contract outside launch_fattn is fragile; any future decode kernel must reuse launch_fattn's alloc/dispatch plumbing (ggml_cuda_flash_attn_ext_get_alloc_size + dst_tmp path), not a hand-rolled launcher. The 37 GB/s VEC wall appears to be a hard structural floor for this fp16 decode shape on Maxwell - 3 independent approaches (knob tuning, split-K, new kernel) all failed to beat it.
4. MoE pp at ub>128: INVESTIGATED - negative result (this session, do not redo). Baseline e2e: pp2048(-ub2048)=360.45, pp8192(-ub8192)=331.47 t/s. Experiment 1: MMVQ_MOE_CHUNK_MAX_BATCH 128->512 gives op-level Q4_K n=256: 374->621, n=512: 701->614 GFLOPS, but e2e pp2048 360->361 (neutral), pp8192 331 (unchanged). Experiment 2: cap 2048 (route ub=2048 fully to fused) REGRESSES pp2048 to 218 t/s. Conclusion: the fused MMVQ MoE path plateaus at the GEMV-with-dequant compute ceiling (~620-850 GFLOPS) while dequant+batched-cuBLAS SGEMM wins above ~512 tokens (amortized expert reads); fallback cost is only ~12% e2e at ub 2048/8192. "Dequant only used experts" rejected on analysis: needs an extra stream sync (compaction) to save ~15% of dequant time in a ~12%-impact path. Fused kernel is hard-capped at chunk=8 on Maxwell (__launch_bounds__ max_batch*32, 1 block/SM). Reverted to MMVQ_MOE_CHUNK_MAX_BATCH=128.
5. CUDA graph warmup-reset on batch-shape changes (chat alternates decode/pp shapes) - flagged earlier, not audited. May matter for tool-use latency.

## Macro knobs in mmvq.cu for future sweeps (edit + rebuild)
`GGML_MMVQ_RPB_MAXWELL_N1` (decode rpb, =4), `GGML_MMVQ_NWARPS_MAXWELL_N1` (=1), `GGML_MMVQ_NWARPS_MAXWELL_N1_IQ` (IQ2/3, =2), `GGML_MMVQ_MOE_RPB_MAXWELL` (MoE rpb, =8), `MMVQ_MOE_CHUNK_MAX_BATCH` (mmvq.cuh, =128).

## Test / benchmark commands
- Correctness: `./build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID,FLASH_ATTN_EXT` (must pass; "not supported" = type-combo skips). Full `test -b CUDA0` has 19 PRE-EXISTING failures (SOFT_MAX/ARGSORT/TOP_K large-array edge cases) confirmed on base 796e62235 - NOT from our work.
- Op perf: `./build/bin/test-backend-ops perf -b CUDA0 -o <OP> > /tmp/x.txt 2>&1`; timing prints on the line AFTER the shape label (graph-warmup logs can land between). Parse: `grep -oE "OP\([^)]*\)|[0-9.]+ GFLOPS|[0-9.]+ TFLOPS" /tmp/x.txt | paste - -`.
- End-to-end: `./build/bin/llama-bench -m ~/gguf/<model> -dev CUDA0 -fa on -ngl 99 -p <P> -ub <UB> -n <N> [-d <ctx>]`.
- Pascal spot-check: `-b CUDA1`.

## Rules (from AGENTS.md, strictly)
- No commit/push/PR-description/reviewer-reply. The 9 commits above were made with user approval via `Assisted-by:`; do NOT push or open PRs (automated submission = project ban).
- ASCII only in code (no unicode arrows/em-dashes). Concise comments. No new subsystems. Match surrounding style. No overengineering / guards against hypotheticals.
- Gate arch-specific changes to cc 5.x (`GGML_CUDA_CC_IS_NVIDIA(cc) && cc>=500 && cc<GGML_CUDA_CC_PASCAL` host pattern, or `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 500 && __CUDA_ARCH__ < GGML_CUDA_CC_PASCAL`). Do not regress Pascal/Volta - spot-check CUDA1.
- GGML_LOG_DEBUG=1 does NOT enable debug graph logs (behind #ifndef NDEBUG) and causes a 127M-line flood + OOM. Do not use. nsys broken. LLAMA_PROFILE removed.
