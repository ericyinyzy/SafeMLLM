#!/usr/bin/env bash
# SafeMLLM evaluation — LLaVA-1.5-13B (defended with SafeMLLM LoRA).
#
# Run from the project root:
#     bash scripts/run_L13B.sh [GPU_ID]
#
# Required env vars:
#     LLAVA13B_BASE  — path to LLaVA-1.5-13B base weights (HF: liuhaotian/llava-v1.5-13b)
#     SAFEMLLM_L13B  — path to the SafeMLLM-trained LoRA for LLaVA-1.5-13B
#                       (HF:   ericyinyzy/SafeMLLM-LLaVA-13B  ← public release)

set -euo pipefail
GPU="${1:-0}"
LLAVA13B_BASE="${LLAVA13B_BASE:?Set LLAVA13B_BASE to the LLaVA-1.5-13b base weight path}"
SAFEMLLM_L13B="${SAFEMLLM_L13B:?Set SAFEMLLM_L13B to the SafeMLLM-trained LoRA path}"

mkdir -p advimg_L13B result_L13B results_figstep results_mmbench results_utility_mmvet

# 1) ImgJP — generate adversarial image and evaluate ASR on AdvBench
CUDA_VISIBLE_DEVICES="$GPU" python attacks/attack_llava_13B.py \
    --model-path  "$SAFEMLLM_L13B" \
    --model-base  "$LLAVA13B_BASE" \
    --advpath     advimg_L13B/L13B_safemllm.pt \
    --index 2 \
    --iters 100

# 2) FigStep
CUDA_VISIBLE_DEVICES="$GPU" python evals/eval_figstep.py \
    --model-path  "$SAFEMLLM_L13B" \
    --model-base  "$LLAVA13B_BASE" \
    --method      'L13B_safemllm'

# 3) MM-SafetyBench
CUDA_VISIBLE_DEVICES="$GPU" python evals/eval_mmbench.py \
    --model-path  "$SAFEMLLM_L13B" \
    --model-base  "$LLAVA13B_BASE" \
    --method      'L13B_safemllm'

# 4) MM-Vet
CUDA_VISIBLE_DEVICES="$GPU" python evals/eval_mmvet.py \
    --model-path  "$SAFEMLLM_L13B" \
    --model-base  "$LLAVA13B_BASE" \
    --outputpath  results_utility_mmvet/L13B_safemllm.json
