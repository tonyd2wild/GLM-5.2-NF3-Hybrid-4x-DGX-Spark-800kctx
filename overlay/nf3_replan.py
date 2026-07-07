"""
NF3 "2+1 split-plane" repack (load-time, VRAM) + expert-space permutation helpers.

Disk checkpoint stays in the native "837" pack (8 codes -> 24-bit LE word -> 3 bytes,
code k = (word >> 3*(k%8)) & 7, uint8 [E, N, K//8*3]).  At load we repack to the
split-plane layout the b12x NF3 kernel format consumes:

  per 32-code scale group -> 12 bytes:
    LO plane: 2 u32 = 32 x 2-bit fields (code j's bits 1:0 at bits 2j+1:2j of the 64-bit lo pair)
    HI plane: 1 u32 = 32 x 1-bit fields (code j's bit 2 at bit j)
  128-code superblock = 48 B = 3 x 16 B cp.async chunks (25% fewer B bytes than 4-bit).
  Rows stay 16B-multiples at K=6144 (2304 B) and K=512 (192 B).

In-kernel selector build for one prmt (4 codes), documented here, implemented in fp4.py:
  lo8  = (lo_u32 >> (8*g)) & 0xFF        # 4 codes' 2-bit fields, group g of 4
  hi4  = (hi_u32 >> (4*g)) & 0xF         # 4 codes' high bits
  # spread2: 4 x 2-bit fields (stride 2) -> nibble slots (stride 4); shift-ladder, carry-safe
  t    = (lo8 | (lo8 << 4)) & 0x0F0F
  s2   = (t   | (t   << 2)) & 0x3333
  # spread1: 4 x 1-bit fields -> nibble slots
  u    = (hi4 | (hi4 << 6)) & 0x0303
  s1   = (u   | (u   << 3)) & 0x1111
  sel  = s2 | (s1 << 2)                  # 4 prmt selector nibbles, each = the 3-bit code
  -> 2 pool-prmt (8 lo / 8 hi codebook bytes in 4 regs) -> 2 interleave-prmt
     (0x5140, 0x7362) -> 2 x bf16x2; then mul by the widened e4m3_k32 scale (bf16x2).

Kernel-facing tensor (per proj): uint32 [E, N, K//32, 3] contiguous, u32[0..1]=LO, u32[2]=HI.
Scales stay checkpoint-native: float8_e4m3fn [E, N, K//32].

NOTE: the intra-group code order (which K element lands at field j) is IDENTITY for now;
if the b12x fragment map consumes K in an interleaved order (k16 two-lane style), set
CODE_ORDER from the K1 read-map so field j = the j-th code the kernel consumes.
"""
import torch

NF3_CODEBOOK = (-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0)
CODE_ORDER = None  # optional [32] permutation: field j holds code group_base + CODE_ORDER[j]


def unpack_837_codes(packed: torch.Tensor, K: int) -> torch.Tensor:
    """uint8 [E, N, K//8*3] (837 layout) -> int32 codes [E, N, K]."""
    E, N, _ = packed.shape
    b = packed.reshape(E, N, K // 8, 3).to(torch.int32)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)          # [E,N,K//8]
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 3
    codes = (word.unsqueeze(-1) >> shifts) & 7                        # [E,N,K//8,8]
    return codes.reshape(E, N, K)


def pack_2p1(codes: torch.Tensor, code_order=None) -> torch.Tensor:
    """int codes [E, N, K] (0..7) -> uint32 [E, N, K//32, 3] split-plane (LO,LO,HI)."""
    E, N, K = codes.shape
    assert K % 32 == 0, K
    c = codes.to(torch.int64).reshape(E, N, K // 32, 32)              # [E,N,G,32]
    if code_order is not None:
        c = c[..., code_order]
    j = torch.arange(32, device=codes.device, dtype=torch.int64)
    lo = c & 3                                                        # bits 1:0
    hi = (c >> 2) & 1                                                 # bit 2
    lo64 = (lo << (2 * j)).sum(-1, dtype=torch.int64)                 # [E,N,G] 64-bit
    lo_w0 = lo64 & 0xFFFFFFFF
    lo_w1 = (lo64 >> 32) & 0xFFFFFFFF
    hi_w = (hi << j).sum(-1, dtype=torch.int64) & 0xFFFFFFFF
    return torch.stack([lo_w0, lo_w1, hi_w], dim=-1).to(torch.uint32)


def unpack_2p1(planes: torch.Tensor, code_order=None) -> torch.Tensor:
    """uint32 [E, N, G, 3] -> int32 codes [E, N, G*32] (inverse of pack_2p1)."""
    E, N, G, _ = planes.shape
    p = planes.to(torch.int64)
    lo64 = p[..., 0] | (p[..., 1] << 32)
    j = torch.arange(32, device=planes.device, dtype=torch.int64)
    lo = (lo64.unsqueeze(-1) >> (2 * j)) & 3                          # [E,N,G,32]
    hi = (p[..., 2].unsqueeze(-1) >> j) & 1
    c = (lo | (hi << 2))
    if code_order is not None:
        inv = torch.empty_like(torch.as_tensor(code_order))
        inv[torch.as_tensor(code_order)] = torch.arange(32)
        c = c[..., inv]
    return c.reshape(E, N, G * 32).to(torch.int32)


def repack_837_to_2p1(packed_837: torch.Tensor, K: int, code_order=None) -> torch.Tensor:
    """One-shot load-time repack, chunked over E to bound peak VRAM."""
    E, N, _ = packed_837.shape
    out = torch.empty(E, N, K // 32, 3, dtype=torch.uint32, device=packed_837.device)
    step = max(1, min(E, 8))
    for e0 in range(0, E, step):
        codes = unpack_837_codes(packed_837[e0:e0 + step], K)
        out[e0:e0 + step] = pack_2p1(codes, code_order)
    return out


def hybrid_expert_permutation(kept_global_ids: torch.Tensor, n_experts: int = 256) -> torch.Tensor:
    """perm[new_id] = old_id with kept (NVFP4) experts first: new 0..63 = kept, 64.. = NF3.

    Apply to: expert weight order, gate.weight rows, e_score_correction_bias entries.
    n_group==1/topk_group==1 on GLM-5.2 -> renumbering is routing-neutral.
    """
    kept = kept_global_ids.to(torch.long)
    mask = torch.ones(n_experts, dtype=torch.bool)
    mask[kept] = False
    rest = torch.arange(n_experts)[mask]
    return torch.cat([kept, rest])


def selector_build_reference(lo8: int, hi4: int) -> int:
    """Host-side reference of the in-kernel selector build (4 codes -> 4 prmt nibbles)."""
    t = (lo8 | (lo8 << 4)) & 0x0F0F
    s2 = (t | (t << 2)) & 0x3333
    u = (hi4 | (hi4 << 6)) & 0x0303
    s1 = (u | (u << 3)) & 0x1111
    return s2 | (s1 << 2)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    import sys
    sys.path.insert(0, "/Users/phantom/Downloads/glm52-hybrid")
    from nf3_kernel import pack_nf3_837
    E, N, K = 3, 8, 128
    codes = torch.randint(0, 8, (E, N, K), device=dev)
    p837 = pack_nf3_837(codes)
    rt = unpack_837_codes(p837, K)
    assert torch.equal(rt, codes.to(torch.int32)), "837 unpack mismatch"
    planes = pack_2p1(codes)
    rt2 = unpack_2p1(planes)
    assert torch.equal(rt2, codes.to(torch.int32)), "2+1 round-trip mismatch"
    planes2 = repack_837_to_2p1(p837, K)
    assert torch.equal(planes2, planes), "837->2p1 repack mismatch"
    print("PACK ROUND-TRIP: PASS")

    # selector build: exhaustive over all 4-code combos (8^4 = 4096)
    for c0 in range(8):
        for c1 in range(8):
            for c2 in range(8):
                for c3 in range(8):
                    quad = [c0, c1, c2, c3]
                    lo8 = sum((c & 3) << (2 * j) for j, c in enumerate(quad))
                    hi4 = sum(((c >> 2) & 1) << j for j, c in enumerate(quad))
                    sel = selector_build_reference(lo8, hi4)
                    for j in range(4):
                        got = (sel >> (4 * j)) & 0xF
                        assert got == quad[j], (quad, j, got)
    print("SELECTOR BUILD: PASS (exhaustive 8^4)")

    # cross-check selector inputs against packed planes
    g = 3  # arbitrary group-of-4 within the 32-code group
    grp = codes[0, 0, :32]
    lo64 = int(planes[0, 0, 0, 0]) | (int(planes[0, 0, 0, 1]) << 32)
    hiw = int(planes[0, 0, 0, 2])
    lo8 = (lo64 >> (8 * g)) & 0xFF
    hi4 = (hiw >> (4 * g)) & 0xF
    sel = selector_build_reference(lo8, hi4)
    for j in range(4):
        assert (sel >> (4 * j)) & 0xF == int(grp[4 * g + j]), "plane->selector mismatch"
    print("PLANE->SELECTOR: PASS")

    kept = torch.randperm(256)[:64]
    perm = hybrid_expert_permutation(kept)
    assert perm.shape == (256,) and len(set(perm.tolist())) == 256
    assert torch.equal(perm[:64], kept.to(torch.long))
    print("PERMUTATION: PASS")
