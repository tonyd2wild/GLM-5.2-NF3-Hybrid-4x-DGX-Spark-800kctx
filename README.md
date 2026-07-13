# GLM-5.2 NF3-Hybrid on 4× DGX Spark — 67.4 tok/s aggregate / 800K context

> Unpruned GLM-5.2 (753B/40B MoE, all 256 experts) in a 3-format NVFP4/NF3/MXFP8 hybrid quant (**327GB**), ported from 4× RTX PRO 6000 to **4× DGX Spark (GB10, sm_121a, aarch64)** and served in two launch lanes: a fast 200K lane (67.4 tok/s aggregate) and an 800K-context lane.

**Status: 🚧 SERVING (2026-07-07) — private until publish.**

First port of [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid) — unpruned GLM-5.2 (753B/40B MoE, all 256 experts) in a 3-format hybrid quant (**327GB**: top-64 experts/layer NVFP4, remaining 192 in custom 3-bit **NF3**, MXFP8 non-expert) — from its native 4× RTX PRO 6000 (sm_120, amd64) to **4× DGX Spark (GB10, sm_121a, aarch64)**.

## TL;DR

- **One model, one image, one set of weights — two launch envs.** Pick a lane per workload; both TP4-shard the model weights and differ only in where the KV cache lives (DCP).
- **🏎️ FAST LANE (DCP1):** max-model-len 200,000, KV pool 219,264 tokens. **24–29 tok/s** single-stream, **67.4 tok/s aggregate @ c6.** For interactive chat, agents, throughput.
- **🧠 CONTEXT LANE (DCP4):** max-model-len 800,000, KV pool **876,588 tokens.** ~19–20 tok/s single-stream. For huge-context jobs and long-doc analysis.
- Tool-calling and reasoning parser are on from first boot in both lanes.

## Hardware

- **4× DGX Spark (GB10, sm_121a, aarch64)** — 48 SMs per GPU; unified memory (author's dedicated-VRAM `gpu_memory_utilization 0.96` is dropped to 0.88 here for GB10 unified memory).
- **Fabric:** each node on the RoCE fabric (`HS_IFACE=enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0`); head `192.168.192.1`, workers `192.168.192.2 192.168.192.3 192.168.192.4`. Edit these for your own network.
- **Disk:** 327GB/node landing zone on all 4 Sparks (334G on disk, 96 files).
- Decode on GB10 is memory-bandwidth-bound (~273GB/s): NF3's 3-bit cold experts move ~25% fewer bytes per token than 4-bit, and the 78GB weight savings goes straight into KV cache.

## Quick start

```bash
# 1. Build the aarch64/sm_121a base image on a GB10 Spark
#    (requires m9e/blackwell-llm-docker checked out at build/spark-vllm-docker)
./build/run-nf3-build.sh            # -> vllm-nf3-hybrid:base-arm64

# 2. Bake MadeBy561's v2 NF3 runtime overlay onto the base image
./build/bake-nf3-overlay.sh         # -> vllm-nf3-hybrid:probe (verifies overlay, prints "OVERLAY VERIFIED OK")

# 3. Download the 327GB checkpoint to every node
huggingface-cli download madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid \
  --local-dir /var/tmp/models/glm52-nf3-hybrid

# 4. Pick a lane, edit its fabric block for your network, and launch from the recipe env:
#    FAST (200K):    recipes/glm52-nf3-dcp1-200k-speed.env
#    CONTEXT (800K): recipes/glm52-nf3-dcp4-800k.env
```

Both lanes serve on `PORT=8210` as `SERVED_MODEL_NAME=glm-5.2`.

## Setup (detailed)

### Weights

- **HF model id:** [`madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid) — **327GB** (334G on disk, 96 files).
- Download to `/var/tmp/models/glm52-nf3-hybrid` on every node (`MODEL_PATH` in the recipe envs), then fabric fan-out (~680MB/s across the 4 Sparks).

### Image / build

Build the base image on a single GB10 Spark (`build/run-nf3-build.sh`, ~1h23m, exit 0), then bake the NF3 runtime overlay (`build/bake-nf3-overlay.sh`). The base build calls `m9e/blackwell-llm-docker`'s `build-and-copy.sh --gpu-arch 12.1a` and must have that repo checked out as the working dir.

Build pins (= the author's v2 image, rebuilt for aarch64/sm_121a):

| component | ref |
|---|---|
| vLLM | `local-inference-lab/vllm` @ `dev/eldritch-enlightenment` `45c1582e9b80ba83e71c3a6458e71da4736fbdc4` |
| b12x | `voipmonitor/b12x` @ `f3686b555d639823b276c2080f173145eed7f007` |
| NF3 kernel + hybrid loader | MadeBy561's v2 runtime overlay (source: `MadeBy561/b12x@nf3-hybrid` + v2 files from his public docker image) |
| flashinfer | 0.6.14 aarch64 wheels (author used 0.6.13; APIs compatible) |
| CUDA | 13.2 / torch 2.12.0+cu132 |
| arch | `TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a` |
| builder | m9e/blackwell-llm-docker (`build-and-copy.sh --gpu-arch 12.1a`) |

The overlay bake (`build/bake-nf3-overlay.sh`) copies the v2 loader/kernels (`hybrid_loader.py`, `nf3_kernel.py`, `nf3_replan.py`, `nvfp4_kernel.py`, `mxfp8_tier.json`) plus the b12x NF3 kernel 4-file diff and the warmup-fallback-patched MLA `indexer.py`, installs `nvidia-cutlass-dsl==4.5.2` + `nvidia-cutlass-dsl-libs-cu13==4.5.2` with the `.pth` path hook, and commits `vllm-nf3-hybrid:probe`.

**Portability facts (recon, 2026-07-07):** sm_120 and sm_121 are the same ISA tier (no tcgen05 on either; NVFP4 block-scaled MMA exists on both per PTX ISA); the NF3 CuteDSL kernel ships an `_W4A16_REGS_SM121` register table — the author tuned it on SM121-class hardware; GB10's ALU-per-byte ratio exceeds the RTX PRO 6000's, so the memory-bound kernel stays memory-bound here.

### Launch

Source the chosen recipe env (`recipes/glm52-nf3-dcp1-200k-speed.env` or `recipes/glm52-nf3-dcp4-800k.env`) — it carries the full launch config: image `vllm-nf3-hybrid:probe`, `MODEL_PATH`, TP/DCP sizing, `QUANTIZATION=modelopt_fp4`, `KV_CACHE_DTYPE=fp8`, the `SPECULATIVE_CONFIG` MTP block, the b12x/hybrid container env, and the fabric block. Set `HEAD_IP`/`WORKER_IPS`/`SSH_KEY`/`HS_IFACE`/`NCCL_IB_HCA` for your actual fabric (defaults point at the recipe author's network; `WORKER_IPS` must be quoted).

### Verify

- `build/bake-nf3-overlay.sh` self-verifies the overlay (imports `hybrid_loader`/`nf3_kernel`/`nf3_replan`/`nvfp4_kernel`, asserts the NF3 kernel overlay is present, checks the indexer block-table fix) and prints `[bake] OVERLAY VERIFIED OK`.
- After launch, tool-calling and reasoning are on from first boot (`REASONING_PARSER=glm45`, `TOOL_CALL_PARSER=glm47`, `ENABLE_AUTO_TOOL_CHOICE=1`); smoke-test the OpenAI-compatible endpoint on `:8210` against model `glm-5.2`.

## Benchmarks

All runs: 512-token gens, temp 0, same bench script as our published repos.

### 🏎️ FAST LANE (DCP1) — measured 2026-07-07

TP4, DCP1, MTP k=4, fp8 KV, max-model-len 200,000, max_num_seqs 6, gmu 0.88, kv-cache-memory-bytes 12e9. **KV pool: 219,264 tokens.**

| streams | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| tok/s | 24–29 | 36.6 | 45.7 | 54.2 | 59.2 | **67.4** |

c1 is content-dependent: MTP acceptance swings 3.3→4.0 (measured runs: 29.1 @ accept 4.02, 24.4 @ 3.29, 23.8 @ 3.26). Beats our published QuantTrio 200K recipe on aggregate (67.4 vs 60.5); c1 comparable (28.8 median there).

### 🧠 CONTEXT LANE (DCP4)

TP4, DCP4 ag_rs interleave 1, MTP k=4 (was 5 at first boot), fp8 KV, max_num_seqs 4. **KV pool: 876,588 tokens** (vs 657K on our previous 655K recipe — +33%).

The 240K/k=5 first-boot shape measured:

| streams | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| tok/s (240K, k=5) | 19.4–19.9 | 30.5 | 35.7 | 42.3 | 36.5 | 34.9 |

Acceptance 3.3–3.7.

**800K CONFIRMED SERVING** — `max_model_len: 800000`, KV pool **877,056 tokens**, boot ~20 min:

| conc | aggregate tok/s | per-stream avg | accept len |
|---|---|---|---|
| c1 | 21.4–21.6 | 21.5 | 3.02 |
| c2 | 32.9 | 16.7 | 3.08 |
| c3 | 38.9 | 13.4 | 2.97 |
| c4 | 45.4 | 11.6 | 2.91 |
| c5 | 37.4 | 10.8 | 2.92 |
| c6 | 40.1 | 9.9 | 2.94 |

(Shallow context, **MTP k=3 + fuse_allreduce_rms** — k=3 with near-perfect acceptance beat k=4/5 by ~8%. Decode speed at 800K max-len matches the 240K shape — the larger ceiling is free at equal fill. Deep-context depth bench: not yet run.)

### Targets vs achieved

| | our QuantTrio recipes (published) | NF3 hybrid target | NF3 hybrid achieved |
|---|---|---|---|
| weights | 405GB (98GB/node) | **327GB (~82GB/node)** | ✅ 327GB (334G on disk) |
| single-stream | 28.8 tok/s (200K) / 23.0 (655K) | ≥30 tok/s | 24–29 tok/s (FAST lane, content-dependent) |
| aggregate | 60.5 tok/s (200K c6) | — | **67.4 tok/s (FAST lane c6)** |
| context | 655,360 tokens (KV pool 657K) | 1M-class pool stretch goal | **KV pool 876,588 tokens; 800K len measured 2026-07-07** |
| MTP | k=3 | k=5 (author's config) | k=4 (accept 3.3–4.0) |

## Configuration

### The two lanes

| | 🏎️ FAST LANE | 🧠 CONTEXT LANE |
|---|---|---|
| env | [`recipes/glm52-nf3-dcp1-200k-speed.env`](recipes/glm52-nf3-dcp1-200k-speed.env) | [`recipes/glm52-nf3-dcp4-800k.env`](recipes/glm52-nf3-dcp4-800k.env) |
| DCP | 1 (KV local per node) | 4 (KV sharded across all 4 nodes) |
| max-model-len | 200,000 | 800,000 |
| KV pool | 219,264 tokens | **876,588 tokens** |
| single-stream | **24–29 tok/s** | ~19–20 tok/s (240K first boot) |
| aggregate | **67.4 tok/s @ c6** | 42.3 @ c4 (240K first boot) |
| use it for | interactive chat, agents, throughput | huge-context jobs, long-doc analysis |

**Why the lanes differ (plain English):** DCP controls where the KV cache lives. **DCP4** shards every sequence's KV across all 4 Sparks → 4× the pool, but every decode step pays cross-node gathers on all 78 layers (ag_rs). **DCP1** keeps each sequence's KV on one node → zero attention network traffic, ~1/4 the pool. Model weights are TP4-sharded in both lanes. Speed is content-dependent under MTP: acceptance rises on predictable text, so expect bursts above the medians.

### Tuning notes (measured, 2026-07-07)

We swept MTP k and the `fuse_allreduce_rms` compile flag on both lanes. Findings:

| lane | best config | why |
|---|---|---|
| CONTEXT (DCP4) | **k=3 + fuse_allreduce_rms** | long network-taxed steps: wasted draft passes hurt (k=3 acceptance ~3.0 is near-perfect), and fusing the per-layer allreduce+norm pays. +7.5% c1 vs k=5 unfused. |
| FAST (DCP1) | **k=4, no fuse** | short local steps: the extra draft token amortizes (k=4 ties k=3 on median, wins peak 29.1 and aggregate 67.4); fuse doesn't help without the DCP network tax. |

**k is lane-specific** — don't copy the speculative config between shapes. FAST-lane c1 medians ~24.4 across all configs; peaks up to 29 on predictable content. Pushing the median past 30 looks like NF3-kernel tuning for GB10's 48 SMs, not launch flags.

### Author's launch recipe (single-box reference, adapted above)

TP4 + DCP4 `ag_rs` interleave 1, `--kv-cache-dtype fp8`, `B12X_MLA_SPARSE`, `moe_backend b12x`, MTP-5 (`{"method":"mtp","num_speculative_tokens":5,"moe_backend":"b12x","draft_sample_method":"probabilistic"}`), gmu 0.96, `--max-model-len 240000`, mnbt 4096, capture 64. Env: `HYBRID_TIER=both HYBRID_KEPT=b12x_nf3 HYBRID_NF3=b12x_nf3 HYBRID_MXFP8_NATIVE=1 B12X_MOE_FORCE_A16=1 B12X_W4A16_TC_DECODE=1 VLLM_DCP_GLOBAL_TOPK=1 VLLM_DCP_SHARD_DRAFT=1`.

Known walls (author's WORKING_CONFIGS): cudagraph capture must cover `seqs×(1+k)`; `--max-num-batched-tokens` ≥2048 or NF3 graph capture breaks; `VLLM_DCP_SHARD_DRAFT=1` mandatory for MTP draft KV under DCP.

## Troubleshooting

Boot fixes — things that WILL bite you:

1. **`HYBRID_MXFP8_TIER_JSON`** must point at the real `mxfp8_tier.json` path — the loader hardcodes the author's `/opt/venv` layout. Symptom: "mxfp8 overlay FAILED" + silent bf16 fallback that eats KV memory. Ours: `/usr/local/lib/python3.12/dist-packages/mxfp8_tier.json`.
2. **`vllm/model_executor/warmup/b12x_sparse_indexer_warmup.py`** must be the author's patched version (try/except ImportError fallback around `fused_indexer_decode_warmup_rows`). The file exists ONLY in his docker image (layer 19), not in any public repo. Symptom: worker crash "cannot import name 'fused_indexer_decode_warmup_rows'" right after KV-pool init.
3. **`nvidia-cutlass-dsl==4.5.2` + `nvidia-cutlass-dsl-libs-cu13==4.5.2` + the `nvidia_cutlass_dsl.pth` path hook** must be installed — the eugr-built base lacks them. Symptom: `ModuleNotFoundError: cutlass`.
4. **Do not reapply our DSA indexer +1 fix** — this vLLM tree (45c1582) already contains the off-by-one fix as "+8 reaper fix".
5. **Launch envs need `HEAD_IP`/`WORKER_IPS`/`SSH_KEY`/`HS_IFACE`/`NCCL_IB_HCA`** set for your actual fabric (defaults point at the recipe author's network), and `WORKER_IPS` must be quoted.

## Credits & links

- **madeby561 (Hunter Wolf)** — the NF3 format, kernel, hybrid loader, and checkpoint. This is his work; we port it.
- **lukealonso** — NVFP4 donor checkpoint + b12x kernel library (SM120/121)
- **voipmonitor / local-inference-lab** — eldritch vLLM/b12x forks and build lineage
- **m9e (Matt) / eugr** — the Spark-native build harness
- **Zatz, CosmicRaisins, ciprianveg** — the GLM-on-Spark foundation this builds on
- Our prior recipes: [200K speed shape](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark) · [655K+MTP shape](https://github.com/tonyd2wild/GLM-5.2-655K-MTP-4x-DGX-Spark)
</content>
</invoke>
