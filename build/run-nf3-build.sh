#!/usr/bin/env bash
# aarch64/sm_121a base image build for the NF3 hybrid stack (runs on a GB10 Spark).
# Requires m9e/blackwell-llm-docker checked out as the working dir.
set -euo pipefail
cd "$(dirname "$0")/spark-vllm-docker"
./build-and-copy.sh \
  --gpu-arch 12.1a \
  -j 8 \
  --vllm-repo https://github.com/local-inference-lab/vllm \
  --vllm-ref 45c1582e9b80ba83e71c3a6458e71da4736fbdc4 \
  --vllm-commit 45c1582e9b80ba83e71c3a6458e71da4736fbdc4 \
  --b12x-repo https://github.com/voipmonitor/b12x \
  --b12x-ref f3686b555d639823b276c2080f173145eed7f007 \
  --b12x-commit f3686b555d639823b276c2080f173145eed7f007 \
  --full-log \
  -t vllm-nf3-hybrid:base-arm64
