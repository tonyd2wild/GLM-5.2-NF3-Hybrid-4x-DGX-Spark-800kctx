"""
NF3 codebook MoE GEMM (Triton) — reads the checkpoint's NATIVE 8->3-byte packing directly
(no repack). W[e,n,k] = NF3_codebook[code] * blockscale[e,n,k//GS]. Packing (matches build's
quant_nvfp3): 8 codes -> 24-bit word -> 3 uint8 bytes; code k = (word >> 3*(k%8)) & 7.
Kernel = sorted-token grouped GEMM (reuses vLLM moe_align_block_size) with inline unpack+dequant.
Importable: nf3_moe_layer(hidden, w13_packed, s13, w2_packed, s2, topk_ids, topk_w, emap, ...).
"""
import torch, triton, triton.language as tl

_NF3 = [-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0]


@triton.jit
def _lut(code):
    return tl.where(code == 0, -1.0,
        tl.where(code == 1, -0.6047,
        tl.where(code == 2, -0.3563,
        tl.where(code == 3, -0.1275,
        tl.where(code == 4, 0.1275,
        tl.where(code == 5, 0.3563,
        tl.where(code == 6, 0.6047, 1.0)))))))


@triton.jit
def nf3_moe_kernel(
    a_ptr, b_ptr, s_ptr, c_ptr, tw_ptr,
    sorted_ids_ptr, expert_ids_ptr, ntpp_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,          # packed weight [E, N, K//8*3]
    stride_se, stride_sn, stride_sk,          # block scale  [E, N, K//GS]
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
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        kk = kb * BLOCK_K + offs_k                          # [BLOCK_K]
        kmask = kk < K
        a = tl.load(a_ptrs, mask=tok_mask[:, None] & (kk[None, :] < K), other=0.0)
        # native 8->3 packing: word = 3 bytes at group g=k//8; code = (word >> 3*(k%8)) & 7
        g = (kk[:, None] // 8)
        sh = (3 * (kk[:, None] % 8)).to(tl.int32)
        wb = e * stride_be + offs_bn[None, :] * stride_bn + (g * 3) * stride_bk
        b0 = tl.load(b_ptr + wb + 0 * stride_bk).to(tl.int32)
        b1 = tl.load(b_ptr + wb + 1 * stride_bk).to(tl.int32)
        b2 = tl.load(b_ptr + wb + 2 * stride_bk).to(tl.int32)
        word = b0 | (b1 << 8) | (b2 << 16)
        code = (word >> sh) & 7                              # [BLOCK_K, BLOCK_N]
        v = _lut(code)
        sb = e * stride_se + offs_bn[None, :] * stride_sn + (kk[:, None] // GS) * stride_sk
        s = tl.load(s_ptr + sb, mask=kmask[:, None], other=0.0).to(tl.float32)
        b = (v * s).to(compute_type)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
    if MUL_W:
        w = tl.load(tw_ptr + offs_token, mask=tok_mask, other=0.0)
        acc = acc * w[:, None]
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, acc.to(compute_type), mask=tok_mask[:, None])


def pack_nf3_837(codes):
    """codes [E,N,K] (0..7) -> packed uint8 [E,N,K//8*3] (build's 8->24bit->3byte layout)."""
    E, N, K = codes.shape
    c = codes.to(torch.int32).reshape(E, N, K // 8, 8)
    word = torch.zeros(E, N, K // 8, dtype=torch.int32, device=codes.device)
    for i in range(8):
        word |= (c[..., i] & 7) << (3 * i)
    b = torch.stack([(word & 0xFF), ((word >> 8) & 0xFF), ((word >> 16) & 0xFF)], -1)
    return b.to(torch.uint8).reshape(E, N, K // 8 * 3)


def ref_dequant(codes, scale, GS):
    nf = torch.tensor(_NF3, device=codes.device)
    return nf[codes.long()] * scale.repeat_interleave(GS, dim=2).float()


def run_kernel(a, b, s, sorted_ids, expert_ids, ntpp, N, K, top_k, GS,
               mul_w=False, tw=None, out=None, BM=16, BN=64, BK=32, warps=4):
    M = a.shape[0]
    c = out if out is not None else torch.zeros(M * top_k, N, dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(sorted_ids.shape[0], BM) * triton.cdiv(N, BN),)
    nf3_moe_kernel[grid](
        a, b, s, c, tw if tw is not None else a,
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


def nf3_moe_layer(hidden, w13_packed, s13, w2_packed, s2, topk_ids, topk_w, emap,
                  Etot, GS, H, I, BM=16):
    """Full NF3 MoE layer -> [M, H]. w13_packed [Km,2I,H//8*3], w2_packed [Km,H,I//8*3]."""
    import torch.nn.functional as F
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    M, tk = topk_ids.shape
    sids, eids, ntpp = moe_align_block_size(topk_ids.to(torch.int32), BM, Etot, expert_map=emap)
    sids, eids, ntpp = sids.to(torch.int32), eids.to(torch.int32), ntpp.to(torch.int32)
    c1 = run_kernel(hidden, w13_packed, s13, sids, eids, ntpp, 2 * I, H, tk, GS)     # [M*tk, 2I]
    g, u = c1.chunk(2, -1)
    c2 = (F.silu(g.float()) * u.float()).to(torch.bfloat16)                          # [M*tk, I]
    c3 = run_kernel(c2, w2_packed, s2, sids, eids, ntpp, H, I, 1, GS,
                    mul_w=True, tw=topk_w.reshape(-1).float())                       # [M*tk, H]
    return c3.view(M, tk, H).sum(1)


if __name__ == "__main__":
    import time
    torch.manual_seed(0); dev = "cuda"
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    GS = 32
    # TEST1: single expert
    E, N, K = 1, 128, 256
    codes = torch.randint(0, 8, (E, N, K), device=dev)
    scale = (torch.rand(E, N, K // GS, device=dev) * 0.05 + 0.01).to(torch.float8_e4m3fn)
    B = pack_nf3_837(codes)
    a = torch.randn(16, K, device=dev, dtype=torch.bfloat16)
    W = ref_dequant(codes, scale, GS)[0]
    ref = (a.float() @ W.t()).to(torch.bfloat16)
    sids = torch.full((16,), 16, dtype=torch.int32, device=dev); sids[:16] = torch.arange(16, device=dev)
    eids = torch.zeros(1, dtype=torch.int32, device=dev); ntpp = torch.tensor([16], dtype=torch.int32, device=dev)
    out = run_kernel(a, B, scale, sids, eids, ntpp, N, K, 1, GS)[:16]
    r1 = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
    print(f"TEST1 single-expert: rel={r1:.5f} {'PASS' if r1 < 0.02 else 'FAIL'}")

    # TEST4: full MoE layer vs reference (the integration unit)
    import torch.nn.functional as F
    Etot, Kept = 256, 64; Km = Etot - Kept
    emap = torch.full((Etot,), -1, dtype=torch.int32, device=dev); emap[Kept:] = torch.arange(Km, dtype=torch.int32, device=dev)
    Hs, Is, tk = 512, 128, 8
    c13 = torch.randint(0, 8, (Km, 2 * Is, Hs), device=dev); sc13 = (torch.rand(Km, 2 * Is, Hs // GS, device=dev) * 0.05 + 0.02).to(torch.float8_e4m3fn)
    c2t = torch.randint(0, 8, (Km, Hs, Is), device=dev); sc2 = (torch.rand(Km, Hs, Is // GS, device=dev) * 0.05 + 0.02).to(torch.float8_e4m3fn)
    B13, B2 = pack_nf3_837(c13), pack_nf3_837(c2t)
    Mt = 8
    hid = torch.randn(Mt, Hs, device=dev, dtype=torch.bfloat16) * 0.5
    tids = torch.randint(0, Etot, (Mt, tk), device=dev, dtype=torch.int32)
    tw = torch.softmax(torch.rand(Mt, tk, device=dev), -1)
    out4 = nf3_moe_layer(hid, B13, sc13, B2, sc2, tids, tw, emap, Etot, GS, Hs, Is)
    W13, W2 = ref_dequant(c13, sc13, GS), ref_dequant(c2t, sc2, GS)
    ref4 = torch.zeros(Mt, Hs, device=dev)
    for m in range(Mt):
        for slot in range(tk):
            gg = int(tids[m, slot])
            if gg < Kept: continue
            l = gg - Kept
            inter = hid[m].float() @ W13[l].t(); a_, b_ = inter.chunk(2, -1)
            ref4[m] += tw[m, slot].float() * ((F.silu(a_) * b_) @ W2[l].t())
    r4 = (out4.float() - ref4).abs().sum() / (ref4.abs().sum() + 1e-6)
    print(f"TEST4 full-MoE-layer: rel={r4:.5f} {'PASS' if r4 < 0.03 else 'FAIL'}")

    # TEST3: speed sweep (per-rank GLM shapes)
    H, I = 6144, 512
    c13b = torch.randint(0, 8, (Km, 2 * I, H), device=dev); s13b = (torch.rand(Km, 2 * I, H // GS, device=dev) * 0.03 + 0.01).to(torch.float8_e4m3fn)
    c2b = torch.randint(0, 8, (Km, H, I), device=dev); s2b = (torch.rand(Km, H, I // GS, device=dev) * 0.03 + 0.01).to(torch.float8_e4m3fn)
    B13b, B2b = pack_nf3_837(c13b), pack_nf3_837(c2b)
    for Mv in [1, 4, 8]:
        hv = torch.randn(Mv, H, device=dev, dtype=torch.bfloat16)
        tidv = torch.randint(Kept, Etot, (Mv, tk), device=dev, dtype=torch.int32)
        twv = torch.rand(Mv, tk, device=dev)
        for _ in range(3): nf3_moe_layer(hv, B13b, s13b, B2b, s2b, tidv, twv, emap, Etot, GS, H, I)
        torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(20 * 75): nf3_moe_layer(hv, B13b, s13b, B2b, s2b, tidv, twv, emap, Etot, GS, H, I)
        e1.record(); torch.cuda.synchronize()
        dt = e0.elapsed_time(e1) / 1000 / 20 / Mv
        print(f"TEST3 M={Mv}: {dt*1000:.2f} ms/tok -> {1/dt:.1f} tok/s (NF3 full-layer incl align, 75L, 1 GPU)")
