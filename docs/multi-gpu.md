# Using multiple GPUs with llama.cpp

This guide covers llama.cpp's split modes, the flags that control them, and the cross-GPU AllReduce backends. The arguments are the same for `llama-cli`, `llama-server`, and the other tools.

---

## Split modes

Set with `--split-mode` / `-sm`.

| Mode | What it does | Requires | Cost |
|---|---|---|---|
| `none` | Model on one GPU, chosen by `--main-gpu`. | one GPU | limited to that GPU's VRAM. |
| `layer` (**default**) | Pipeline parallelism: each GPU owns a contiguous slice of layers and the KV for layer *l* stays on the GPU that owns it. No cross-GPU reductions. | nothing special | each GPU holds its own KV; token generation is bounded by the longest pipeline stage. |
| `row` | Legacy row-split path; splits dense weights only. Superseded by both `layer` and `tensor`. | - | no reason to choose it in new deployments. |
| `tensor` | **Experimental.** Tensor parallelism: weights and KV split across all visible GPUs via a meta device. | flash attention; unquantized KV; architecture on the tensor allow-list | one cross-GPU all-reduce per layer; interconnect-bound. |
| `layer-tensor` | **Experimental.** Hierarchical split: `layer` across groups, `tensor` within each group. Group size set by `-tgs`. | everything `tensor` needs, plus `n_devices % tgs == 0`, `tgs >= 2`, and at least 2 groups. | reduces only within a group, so slow inter-group links are not on the reduction path. |

`tensor` and `layer-tensor` share these hard restrictions:

- Flash attention must be on (`-fa 1`; `-fa auto` resolving to off is a startup error).
- KV cache must be `f32`, `f16`, or `bf16`; quantized KV is not implemented.
- The architecture must be on the tensor allow-list, otherwise startup fails with *"tensor split mode not implemented for architecture 'X'"*. Generally excluded: MoE / hybrid and SSM / RWKV-style architectures.
- `--fit` is not implemented (see the flags table).
- `--tensor-split` is rejected by `layer-tensor`.

---

## Which mode to use

```
1 GPU                                                       -> -sm none
2 GPUs, or slow / no-P2P multi-GPU                          -> the mode that gives 2-device groups:
    exactly 2 GPUs                                             -sm tensor
    >= 4 GPUs                                                  -sm layer-tensor -tgs 2
fast pairwise GPU interconnect (P2P / NVLink)               -> -sm tensor
anything else                                               -> -sm layer
```

Reasoning: with no P2P, every cross-GPU reduction is staged through host memory, so the reduction backend dominates. A 2-device group selects the in-tree `internal` reducer; groups of three or more fall back to the generic `butterfly` reducer (below). `layer` performs no cross-GPU reduction at all and is the safe fallback; on a fast link, `tensor` is the fastest.

---

## Flags

| Short | Long | Value | Default | Notes |
|---|---|---|---|---|
| `-sm` | `--split-mode` | `none`\|`layer`\|`row`\|`tensor`\|`layer-tensor` | `layer` | See above. |
| `-tgs` | `--tensor-group-size` | integer | `0` (unset) | GPUs per tensor group in `layer-tensor` (the flag is `-tgs`, not `-tsg`). Must be `>= 2`, divide the visible device count, and leave at least 2 groups. Groups stride the device list: group *g* = devices *g, g+G, g+2G, ...* with *G = n_devices / tgs*. All groups are uniform. |
| `-ts` | `--tensor-split` | proportions, e.g. `3,1` | mode-dependent | Per-GPU share. `layer` / `row`: memory-proportional when unset. `tensor`: sizes the in-group tensor segments; even when unset. Rejected by `layer-tensor`. |
| `-mg` | `--main-gpu` | device index | `0` | GPU used by `-sm none`; with `row`, the GPU holding intermediate results and KV. Warned and ignored by `layer-tensor`. |
| `-ngl` | `--gpu-layers` / `--n-gpu-layers` | integer \| `auto` \| `all` | `auto` | Layers kept in VRAM. `99` or `all` offloads everything possible. |
| `-fa` | `--flash-attn` | `on`\|`off`\|`auto` | `auto` | Required by `tensor` / `layer-tensor`. |
| `-ctk`, `-ctv` | `--cache-type-k` / `--cache-type-v` | `f32`\|`f16`\|`bf16`\|`q8_0`\|... | `f16` | KV cache types. |
| `-ot` | `--override-tensor` | `pattern=buffer,...` | - | Force a tensor-name pattern onto a buffer type, e.g. keep some weights on CPU to rebalance VRAM across GPUs. |
| `-fit` | `--fit` | `on`\|`off` | `on` | Auto-size unset arguments to device memory. Implemented only for `none` / `layer` / `row`; with `tensor` / `layer-tensor` it aborts and the model loads with the supplied/default arguments instead. Use `-fit off` to silence the warning. |
| `-dev` | `--device` | device names, or `none` | auto | Restrict which devices llama.cpp may use; inspect with `--list-devices`. |
| | `--list-devices` | - | - | Print devices and their memory. |

`CUDA_VISIBLE_DEVICES` controls which GPUs the CUDA backend sees at all; `--device` then selects among them.

---

## Cross-GPU AllReduce (tensor and layer-tensor only)

`layer` and `row` do not reduce across GPUs, so none of this applies to them.

**Automatic policy.** When `GGML_CUDA_ALLREDUCE` is unset: if no pair of participating devices has CUDA peer-to-peer (P2P) access, the in-tree reducer is used - `internal` for a 2-device group, `butterfly` otherwise. If any pair has P2P, the platform default is `nccl` on Linux and `internal` elsewhere. The choice is logged at startup as `ggml_cuda_allreduce: using ...`.

**`GGML_CUDA_ALLREDUCE={nccl|internal|butterfly}`** overrides the policy; when set it always wins. `none` is a deprecated alias for `butterfly`.

- `nccl`: requires a build with NCCL support (`-DGGML_CUDA_NCCL=ON`, the default) and a libnccl install - NCCL is not bundled with CUDA.
- `internal`: llama.cpp's own AllReduce, staged through pinned host memory when P2P is unavailable. **Only initialises for exactly 2-device groups**; larger groups fall back to `butterfly`. Works on compute capability 5.0 and newer.
- `butterfly`: the meta-backend's generic butterfly AllReduce (slowest).

**Internal-reducer tuning:**

- `GGML_CUDA_AR_COPY_THRESHOLD` (default `1048576`, 1 MiB): tensors at least this large use the copy-engine path (chunked overlapped D2H/H2D copies); smaller tensors use the kernel path. `0` disables the copy-engine path.
- `GGML_CUDA_AR_COPY_CHUNK_BYTES`: fixed per-chunk copy size on the copy-engine path. Unset uses `clamp(nbytes/4, 512 KiB, 2 MiB)`; values below 256 KiB are clamped up.
- `GGML_CUDA_AR_BF16_THRESHOLD` (default `131072`, 128 KiB): F32 tensors at least this large travel a BF16 wire; smaller tensors keep the F32 wire. `0` disables the BF16 wire.
- `GGML_CUDA_AR_Q8_0_THRESHOLD`, `GGML_CUDA_AR_Q5_0_THRESHOLD`, `GGML_CUDA_AR_Q4_0_THRESHOLD` (default `0`, disabled): F32 tensors at least the threshold travel the narrower `Q8_0` / `Q5_0` / `Q4_0` wire. When several are enabled the most aggressive satisfied one wins (`Q4_0` > `Q5_0` > `Q8_0` > `BF16` > `F32`). The element count must be rotatable by a Walsh-Hadamard transform. Block-quantized wires are summed only by the in-tree reducer - `internal` (2-device copy path) or the N-GPU quantized ring/butterfly; NCCL cannot sum them.
- `GGML_CUDA_AR_NCCL_P2P`: run the quantized ring's transfers as `ncclSend`/`ncclRecv` pairs (device-to-device). Default is staged through pinned host memory, the reliable path on host-staged / no-P2P rigs.

**Other environment variables:**

- `GGML_CUDA_SPIN_WAIT`: by default CUDA synchronisation uses blocking-sync (low CPU). Set to any value to restore spin-waiting for lower sync/wake latency at the cost of busy-waiting CPU.
- `GGML_CUDA_P2P`: opt-in CUDA peer-to-peer access (any value enables it). Requires driver support and can be unstable on some motherboards/BIOS configurations (e.g. IOMMU enabled). Not needed on a rig whose GPUs report no peer access to begin with.

> **NCCL 8-rank fault.** On an 8-GPU no-P2P rig (observed with libnccl 2.22.3 over PCIe gen1), NCCL's default channel geometry crashes inside `ncclGroupEnd()` at exactly 8 ranks (2-7 ranks are fine). Workaround: `NCCL_MAX_NCHANNELS=1` or `NCCL_BUFFSIZE=512KiB` or less. The automatic policy above never selects NCCL when there is no P2P, so the fault is only hit if `GGML_CUDA_ALLREDUCE=nccl` is forced.

---

## Recommended configuration: no-P2P PCIe multi-GPU

This applies to any multi-GPU box where `nvidia-smi topo -m` shows no `NV#` and `cudaDeviceCanAccessPeer` reports no pairs - i.e. every cross-GPU transfer is staged through host memory, including boxes whose GPUs only share PCIe host bridges (PIX groups).

```bash
llama-cli -m model.gguf -ngl 99 -sm layer-tensor -tgs 2 -fa 1 -ctk f16 -ctv f16
```

With 8 visible devices this forms 4 tensor groups of 2 - `{0,4}, {1,5}, {2,6}, {3,7}` under the default enumeration, each spanning both host bridges - and the automatic policy selects the in-tree `internal` reducer for every group.

Measured on an 8-GPU PCIe-gen1 box with no P2P (173-token prompt, 32 generated tokens, all 8 GPUs, 27B hybrid-SSM model, `f16` KV):

| `-sm` | AllReduce | prompt t/s | eval t/s | peak VRAM/GPU | exit |
|---|---|---|---|---|---|
| `layer` | none (no reduction) | 5.97 | 4.63 | ~4.9 GiB | 0 |
| `tensor` | butterfly | 5.37 | 2.76 | ~0.34 GiB | 0 |
| `layer-tensor -tgs 2` | internal (x4) | 14.59 | 6.92 | ~1.8 GiB | 0 |
| `layer-tensor -tgs 4` | butterfly (x2) | 10.67 | 4.80 | ~0.88 GiB | 0 |

Commands: `llama-cli -m <model> -ngl 99 -fa 1 -ctk f16 -ctv f16 --single-turn -n 32 -p "<prompt>" <split args> --verbose`. Peak VRAM is the sum of the log's model / KV / RS / compute buffer sizes; a tensor-mode group buffer is divided by the group size.

`layer-tensor -tgs 2` wins because it is the only configuration that both keeps every reduction a 2-device `internal` reduction and spreads layers across all GPUs. Pure `tensor` and `tgs 4` use the slower butterfly; pure `layer` leaves most GPUs' compute idle during token generation.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| *"SPLIT_MODE_TENSOR requires flash_attn to be enabled"* | Add `-fa 1`. |
| *"simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented"* | Use `-ctk f16 -ctv f16` (or `bf16` / `f32`). |
| *"tensor split mode not implemented for architecture 'X'"* | Architecture is not on the tensor allow-list; use `-sm layer`. |
| *"number of devices is not divisible by the tensor group size"* | Adjust `-tgs` or `CUDA_VISIBLE_DEVICES` so `n_devices % tgs == 0`. |
| *"needs at least 2 groups"* | Lower `-tgs` or use `-sm tensor` for a single group. |
| *"tensor_split is not supported with LLAMA_SPLIT_MODE_LAYER_TENSOR"* | Drop `-ts`; `layer-tensor` groups are uniform. |
| `--fit` warning in `tensor` / `layer-tensor` | Expected: auto-fit is unimplemented there. Pass `-fit off`, or set `-c` / `-ngl` manually. |
| OOM at startup or prefill in `tensor` / `layer-tensor` | Auto-fit is off, so size memory yourself: lower `-c`, lower `-np` for `llama-server`, or lower `-ngl` as a last resort. |
| Multi-GPU slower than single-GPU | Follow *Which mode to use*; check the `ggml_cuda_allreduce: using ...` line to confirm the intended reducer. |
| GPU not used | `-ngl` too low; try `-ngl 99` or `all`. |
| Instability after `GGML_CUDA_P2P` | Unset it; some motherboards/BIOS do not support P2P reliably. |
