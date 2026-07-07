"""NVFP4 codebook MoE GEMM (Triton) — reads the checkpoint's NATIVE 2-fp4/byte packing DIRECTLY
(no convert/prepare/compaction). W[e,n,k] = e2m1(nibble) * block_scale[e,n,k//16] * global[e].
Sibling of nf3_kernel: same sorted-token grouped GEMM (reuses vLLM moe_align_block_size) with inline
unpack + dequant. Graph-safe. Importable: nvfp4_moe_layer(hidden, w13,s13,g13, w2,s2,g2,
topk_ids, topk_w, emap, Etot, H, I). nibble/e2m1 convention matches vLLM dequantize_to_dtype(swizzle=False)."""
import torch, triton, triton.language as tl


@triton.jit
def _e2m1_mag(mag):
    """3-bit magnitude index -> e2m1 magnitude [0,.5,1,1.5,2,3,4,6] (binary-tree, matches vLLM)."""
    b2 = (mag >> 2) & 1
    b1 = (mag >> 1) & 1
    b0 = mag & 1
    low = tl.where(b1 == 1, tl.where(b0 == 1, 1.5, 1.0), tl.where(b0 == 1, 0.5, 0.0))
    high = tl.where(b1 == 1, tl.where(b0 == 1, 6.0, 4.0), tl.where(b0 == 1, 3.0, 2.0))
    return tl.where(b2 == 1, high, low)


@triton.jit
def nvfp4_moe_kernel(
    a_ptr, b_ptr, s_ptr, g_ptr, c_ptr, tw_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,          # packed weight [E, N, K//2]  uint8 (2 fp4/byte)
    stride_se, stride_sn, stride_sk,          # block scale  [E, N, K//GS] fp8_e4m3
    stride_cm, stride_cn,
    top_k: tl.constexpr, MUL_W: tl.constexpr, GS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_ids_ptr + offs_tid)
    tok_mask = offs_token < num_valid_tokens
    e = tl.load(expert_ids_ptr + pid_m)
    if e == -1:
        return
    g = tl.load(g_ptr + e).to(tl.float32)                 # per-expert global scale
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        kk = kb * BLOCK_K + offs_k                          # [BLOCK_K]
        kmask = kk < K
        a = tl.load(a_ptrs, mask=tok_mask[:, None] & (kk[None, :] < K), other=0.0)
        # nvfp4: 2 fp4/byte. byte at k//2; low nibble = even col, high nibble = odd col.
        byte = tl.load(b_ptr + e * stride_be + offs_bn[None, :] * stride_bn
                       + (kk[:, None] // 2) * stride_bk).to(tl.int32)   # [BLOCK_K, BLOCK_N]
        sh = (4 * (kk[:, None] % 2)).to(tl.int32)
        nib = (byte >> sh) & 0xF
        sign = (nib >> 3) & 1
        mag = nib & 0x7
        v = tl.where(sign == 1, -_e2m1_mag(mag), _e2m1_mag(mag))
        sb = e * stride_se + offs_bn[None, :] * stride_sn + (kk[:, None] // GS) * stride_sk
        s = tl.load(s_ptr + sb, mask=kmask[:, None], other=0.0).to(tl.float32)
        b = (v * s * g).to(compute_type)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
    if MUL_W:
        w = tl.load(tw_ptr + offs_token, mask=tok_mask, other=0.0)
        acc = acc * w[:, None]
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, acc.to(compute_type), mask=tok_mask[:, None])


_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def ref_dequant(weight_u8, bscale, gscale, GS):
    """[E,N,K//2] u8 + [E,N,K//GS] fp8 + [E] f32 -> [E,N,K] f32 (matches dequantize_to_dtype)."""
    E, N, Kp = weight_u8.shape
    K = Kp * 2
    lut = torch.tensor(_E2M1, device=weight_u8.device)
    low = (weight_u8 & 0xF).to(torch.int64)
    high = ((weight_u8 >> 4) & 0xF).to(torch.int64)
    def deq(nib):
        return lut[nib & 0x7] * torch.where((nib >> 3) & 1 == 1, -1.0, 1.0)
    w = torch.stack([deq(low), deq(high)], -1).reshape(E, N, K)     # interleave low,high
    s = bscale.float().repeat_interleave(GS, dim=2)
    return w * s * gscale.float().view(E, 1, 1)


def run_kernel(a, b, s, g, sorted_ids, expert_ids, ntpp, N, K, top_k, GS,
               mul_w=False, tw=None, out=None, BM=16, BN=64, BK=32, warps=4):
    M = a.shape[0]
    c = out if out is not None else torch.zeros(M * top_k, N, dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(sorted_ids.shape[0], BM) * triton.cdiv(N, BN),)
    nvfp4_moe_kernel[grid](
        a, b, s, g, c, tw if tw is not None else a,
        sorted_ids, expert_ids, ntpp,
        N, K, sorted_ids.shape[0], M * top_k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        s.stride(0), s.stride(1), s.stride(2),
        c.stride(0), c.stride(1),
        top_k=top_k, MUL_W=mul_w, GS=GS, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        compute_type=tl.bfloat16, num_warps=warps,
    )
    return c


def nvfp4_moe_layer(hidden, w13, s13, g13, w2, s2, g2, topk_ids, topk_w, emap,
                    Etot, H, I, GS=16, BM=16):
    """Full NVFP4 MoE layer -> [M,H]. w13 [Km,2I,H//2] u8, s13 [Km,2I,H//GS] fp8, g13 [Km] f32."""
    import torch.nn.functional as F
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    M, tk = topk_ids.shape
    sids, eids, ntpp = moe_align_block_size(topk_ids.to(torch.int32), BM, Etot, expert_map=emap)
    sids, eids, ntpp = sids.to(torch.int32), eids.to(torch.int32), ntpp.to(torch.int32)
    c1 = run_kernel(hidden, w13, s13, g13, sids, eids, ntpp, 2 * I, H, tk, GS)        # [M*tk, 2I]
    gg, uu = c1.chunk(2, -1)
    c2 = (F.silu(gg.float()) * uu.float()).to(torch.bfloat16)                          # [M*tk, I]
    c3 = run_kernel(c2, w2, s2, g2, sids, eids, ntpp, H, I, 1, GS,
                    mul_w=True, tw=topk_w.reshape(-1).float())                         # [M*tk, H]
    return c3.view(M, tk, H).sum(1)


if __name__ == "__main__":
    import torch.nn.functional as F
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    torch.manual_seed(0); dev = "cuda"; GS = 16

    # TEST1: single expert vs reference dequant
    E, N, K = 1, 128, 256
    wu8 = torch.randint(0, 256, (E, N, K // 2), device=dev, dtype=torch.uint8)
    bs = (torch.rand(E, N, K // GS, device=dev) * 200 + 1).to(torch.float8_e4m3fn)
    gs = (torch.rand(E, device=dev) * 1e-4 + 1e-5).to(torch.float32)
    a = torch.randn(16, K, device=dev, dtype=torch.bfloat16)
    W = ref_dequant(wu8, bs, gs, GS)[0]
    ref = (a.float() @ W.t()).to(torch.bfloat16)
    sids = torch.arange(16, dtype=torch.int32, device=dev)
    eids = torch.zeros(1, dtype=torch.int32, device=dev); ntpp = torch.tensor([16], dtype=torch.int32, device=dev)
    out = run_kernel(a, wu8, bs, gs, sids, eids, ntpp, N, K, 1, GS)[:16]
    r1 = (out.float() - ref.float()).abs().sum() / (ref.float().abs().sum() + 1e-9)
    print(f"TEST1 single-expert: rel={r1:.5f} {'PASS' if r1 < 0.03 else 'FAIL'}")

    # TEST4: full MoE layer (kept-tier shape) vs reference
    Etot, Kept = 256, 64; Km = Etot - Kept
    emap = torch.full((Etot,), -1, dtype=torch.int32, device=dev)   # -1 = skip (moe_align convention)
    emap[:Kept] = torch.arange(Kept, dtype=torch.int32, device=dev)  # kept globals [0,64)->local
    H, I, tk, M = 512, 128, 8, 8
    Kn = Kept
    w13 = torch.randint(0, 256, (Kn, 2 * I, H // 2), device=dev, dtype=torch.uint8)
    s13 = (torch.rand(Kn, 2 * I, H // GS, device=dev) * 200 + 1).to(torch.float8_e4m3fn)
    g13 = (torch.rand(Kn, device=dev) * 1e-4 + 1e-5).to(torch.float32)
    w2 = torch.randint(0, 256, (Kn, H, I // 2), device=dev, dtype=torch.uint8)
    s2 = (torch.rand(Kn, H, I // GS, device=dev) * 200 + 1).to(torch.float8_e4m3fn)
    g2 = (torch.rand(Kn, device=dev) * 1e-4 + 1e-5).to(torch.float32)
    hid = torch.randn(M, H, device=dev, dtype=torch.bfloat16) * 0.5
    tids = torch.randint(0, Etot, (M, tk), device=dev, dtype=torch.int32)
    tw = torch.softmax(torch.rand(M, tk, device=dev), -1)
    out4 = nvfp4_moe_layer(hid, w13, s13, g13, w2, s2, g2, tids, tw, emap, Etot, H, I, GS)
    W13, W2 = ref_dequant(w13, s13, g13, GS), ref_dequant(w2, s2, g2, GS)
    ref4 = torch.zeros(M, H, device=dev)
    for m in range(M):
        for slot in range(tk):
            e = int(tids[m, slot])
            if e >= Kept:
                continue
            inter = hid[m].float() @ W13[e].t(); ga, ub = inter.chunk(2, -1)
            ref4[m] += tw[m, slot].float() * ((F.silu(ga) * ub) @ W2[e].t())
    r4 = (out4.float() - ref4).abs().sum() / (ref4.abs().sum() + 1e-9)
    print(f"TEST4 full-MoE-layer: rel={r4:.5f} {'PASS' if r4 < 0.03 else 'FAIL'}")
