#!/usr/bin/env bash
# SafeMLLM evaluation — LLaVA-1.5-7B (defended with SafeMLLM LoRA).
#
# Run from the project root (the directory that contains attacks/, evals/, dataset/ ...):
#     bash scripts/run_L7B.sh [GPU_ID]
#
# Required env vars (export them before running, or override on the command line):
#     LLAVA7B_BASE   — path to the LLaVA-1.5-7B base weights (HF: liuhaotian/llava-v1.5-7b)
#     SAFEMLLM_L7B   — path to the SafeMLLM-trained LoRA adapter for LLaVA-1.5-7B
#                      (HF:    ericyinyzy/SafeMLLM-LLaVA-7B  ← public release)

set -euo pipefail
GPU="${1:-0}"
LLAVA7B_BASE="${LLAVA7B_BASE:?Set LLAVA7B_BASE to the LLaVA-1.5-7b base weight path}"
SAFEMLLM_L7B="${SAFEMLLM_L7B:?Set SAFEMLLM_L7B to the SafeMLLM-trained LoRA path}"

mkdir -p advimg_L7B result_L7B results_figstep results_mmbench results_utility_mmvet

# 1) ImgJP — generate adversarial image and evaluate ASR on AdvBench
CUDA_VISIBLE_DEVICES="$GPU" python attacks/attack_llava_7B.py \
    --model-path  "$SAFEMLLM_L7B" \
    --model-base  "$LLAVA7B_BASE" \
    --advpath     advimg_L7B/L7B_safemllm.pt \
    --index 2 \
    --iters 100

# 2) FigStep — black-box typographic-image jailbreak benchmark
CUDA_VISIBLE_DEVICES="$GPU" python evals/eval_figstep.py \
    --model-path  "$SAFEMLLM_L7B" \
    --model-base  "$LLAVA7B_BASE" \
    --method      'L7B_safemllm'

# 3) MM-SafetyBench — multimodal harmful-instruction benchmark
CUDA_VISIBLE_DEVICES="$GPU" python evals/eval_mmbench.py \
    --model-path  "$SAFEMLLM_L7B" \
    --model-base  "$LLAVA7B_BASE" \
    --method      'L7B_safemllm'

# 4) MM-Vet — capability / utility benchmark (verify defense doesn't break the model)
CUDA_VISIBLE_DEVICES="$GPU" python evals/eval_mmvet.py \
    --model-path  "$SAFEMLLM_L7B" \
    --model-base  "$LLAVA7B_BASE" \
    --outputpath  results_utility_mmvet/L7B_safemllm.json
