# GLM-5.2 NF3-Hybrid on 4× DGX Spark — WORK IN PROGRESS

**Status: 🚧 BUILD PHASE (2026-07-07) — private until first successful serve + bench.**

First port of [madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid](https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid) — unpruned GLM-5.2 (753B/40B MoE, all 256 experts) in a 3-format hybrid quant (**327GB**: top-64 experts/layer NVFP4, remaining 192 in custom 3-bit **NF3**, MXFP8 non-expert) — from its native 4× RTX PRO 6000 (sm_120, amd64) to **4× DGX Spark (GB10, sm_121a, aarch64)**.

## Why (targets)

| | our QuantTrio recipes (published) | NF3 hybrid target |
|---|---|---|
| weights | 405GB (98GB/node) | **327GB (~82GB/node)** |
| single-stream | 28.8 tok/s (200K) / 23.0 (655K) | **≥30 tok/s** |
| context | 655,360 tokens (KV pool 657K) | **as much as fits — 1M-class pool is the stretch goal** |
| MTP | k=3 | k=5 (author's config) |

Decode on GB10 is memory-bandwidth-bound (~273GB/s): NF3's 3-bit cold experts move ~25% fewer bytes per token than 4-bit, and the 78GB weight savings goes straight into KV cache.

## Build pins (= the author's v2 image, rebuilt for aarch64/sm_121a)

| component | ref |
|---|---|
| vLLM | `local-inference-lab/vllm` @ `dev/eldritch-enlightenment` `45c1582e9b80ba83e71c3a6458e71da4736fbdc4` |
| b12x | `voipmonitor/b12x` @ `f3686b555d639823b276c2080f173145eed7f007` |
| NF3 kernel + hybrid loader | MadeBy561's v2 runtime overlay (source: `MadeBy561/b12x@nf3-hybrid` + v2 files from his public docker image) |
| flashinfer | 0.6.14 aarch64 wheels (author used 0.6.13; APIs compatible) |
| CUDA | 13.2 / torch 2.12.0+cu132 |
| arch | `TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a` |
| builder | m9e/blackwell-llm-docker (`build-and-copy.sh --gpu-arch 12.1a`) |

Key portability facts (recon, 2026-07-07): sm_120 and sm_121 are the same ISA tier (no tcgen05 on either; NVFP4 block-scaled MMA exists on both per PTX ISA); the NF3 CuteDSL kernel ships an `_W4A16_REGS_SM121` register table — the author tuned it on SM121-class hardware; GB10's ALU-per-byte ratio exceeds the RTX PRO 6000's, so the memory-bound kernel stays memory-bound here.

## Plan / progress

- [x] Recon: kernel source + full v2 runtime recovered from public sources (4-agent sweep)
- [x] Fleet disk cleared (327GB/node landing zone on all 4 Sparks)
- [ ] **← HERE:** aarch64 base image build on Asusi (`vllm-nf3-hybrid:base-arm64`)
- [ ] NF3 v2 overlay + `.pth` hook + indexer fixes → `vllm-nf3-hybrid:probe`
- [ ] 327GB checkpoint download (running on Spark4) + 200G-fabric fan-out
- [ ] Single-node smoke → 4-node Ray TP4+DCP4 launch (adapt author's single-box compose to multi-node)
- [ ] MTP-5 + tool-calling (glm47) + reasoning (glm45) from FIRST boot — no bare launches
- [ ] Bench c1–c6 + max-context probe; beat 30 tok/s
- [ ] Publish (this repo goes public on success)

## Author's launch recipe (single-box reference, to be adapted)

TP4 + DCP4 `ag_rs` interleave 1, `--kv-cache-dtype fp8`, `B12X_MLA_SPARSE`, `moe_backend b12x`, MTP-5 (`{"method":"mtp","num_speculative_tokens":5,"moe_backend":"b12x","draft_sample_method":"probabilistic"}`), gmu 0.96, `--max-model-len 240000`, mnbt 4096, capture 64. Env: `HYBRID_TIER=both HYBRID_KEPT=b12x_nf3 HYBRID_NF3=b12x_nf3 HYBRID_MXFP8_NATIVE=1 B12X_MOE_FORCE_A16=1 B12X_W4A16_TC_DECODE=1 VLLM_DCP_GLOBAL_TOPK=1 VLLM_DCP_SHARD_DRAFT=1`.

Known walls (author's WORKING_CONFIGS): cudagraph capture must cover `seqs×(1+5)`; `--max-num-batched-tokens` ≥2048 or NF3 graph capture breaks; `VLLM_DCP_SHARD_DRAFT=1` mandatory for MTP draft KV under DCP.

## Credits

- **madeby561 (Hunter Wolf)** — the NF3 format, kernel, hybrid loader, and checkpoint. This is his work; we port it.
- **lukealonso** — NVFP4 donor checkpoint + b12x kernel library (SM120/121)
- **voipmonitor / local-inference-lab** — eldritch vLLM/b12x forks and build lineage
- **m9e (Matt) / eugr** — the Spark-native build harness
- **Zatz, CosmicRaisins, ciprianveg** — the GLM-on-Spark foundation this builds on
- Our prior recipes: [200K speed shape](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark) · [655K+MTP shape](https://github.com/tonyd2wild/GLM-5.2-655K-MTP-4x-DGX-Spark)
