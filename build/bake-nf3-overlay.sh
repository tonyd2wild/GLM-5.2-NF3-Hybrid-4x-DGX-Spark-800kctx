#!/usr/bin/env bash
# Bake MadeBy561 v2 NF3 runtime overlay onto the freshly built base image.
set -euo pipefail
BASE="${1:-vllm-nf3-hybrid:base-arm64}"
OUT="${2:-vllm-nf3-hybrid:probe}"
OV=~/nf3-overlay/extracted/opt/venv/lib/python3.12/site-packages

echo "[bake] locating site-packages in $BASE..."
SITE=$(docker run --rm --entrypoint python3 "$BASE" -c "import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))")
echo "[bake] SITE=$SITE"

ENTRY=$(docker inspect "$BASE" --format "{{json .Config.Entrypoint}}")
CMD=$(docker inspect "$BASE" --format "{{json .Config.Cmd}}")
echo "[bake] preserving ENTRYPOINT=$ENTRY CMD=$CMD"

docker rm -f nf3bake 2>/dev/null || true
docker create --name nf3bake --entrypoint sleep "$BASE" infinity >/dev/null

# runtime loader + kernels (v2, extracted from madeby561/vllm-glm52-nvfp4-nf3-hybrid:v2)
for f in hybrid_loader.py nf3_kernel.py nf3_replan.py nvfp4_kernel.py mxfp8_tier.json; do
  docker cp "$OV/$f" "nf3bake:$SITE/$f"
done
printf "import hybrid_loader\n" > /tmp/hybrid.pth
docker cp /tmp/hybrid.pth "nf3bake:$SITE/hybrid.pth"

# b12x NF3 kernel 4-file diff
docker cp "$OV/b12x/moe/fused/w4a16/kernel.py"  "nf3bake:$SITE/b12x/moe/fused/w4a16/kernel.py"
docker cp "$OV/b12x/moe/fused/w4a16/prepare.py" "nf3bake:$SITE/b12x/moe/fused/w4a16/prepare.py"
docker cp "$OV/b12x/moe/fused/w4a16/host.py"    "nf3bake:$SITE/b12x/moe/fused/w4a16/host.py"
docker cp "$OV/b12x/cute/fp4.py"                "nf3bake:$SITE/b12x/cute/fp4.py"

# his warmup-fallback-patched MLA indexer (v2)
docker cp "$OV/vllm/v1/attention/backends/mla/indexer.py" "nf3bake:$SITE/vllm/v1/attention/backends/mla/indexer.py"

docker commit \
  --change "ENTRYPOINT $ENTRY" \
  --change "CMD $CMD" \
  nf3bake "$OUT" >/dev/null
docker rm -f nf3bake >/dev/null

echo "[bake] verifying $OUT..."
docker run --rm --entrypoint python3 "$OUT" - <<'PY'
import importlib.util as u, sys
for m in ("hybrid_loader","nf3_kernel","nf3_replan","nvfp4_kernel"):
    assert u.find_spec(m), f"missing {m}"
import b12x.moe.fused.w4a16.kernel as k
src = open(k.__file__).read()
assert "nf3_2p1" in src and "_W4A16_REGS_SM121" in src, "NF3 kernel overlay missing"
import vllm.v1.attention.backends.mla.indexer as ix
isrc = open(ix.__file__).read()
print("indexer block-table +1 present:", ") + 1" in isrc and "max_num_blocks_per_req" in isrc)
print("[bake] OVERLAY VERIFIED OK")
PY
echo "[bake] DONE -> $OUT"
