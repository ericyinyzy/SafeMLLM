# SafeMLLM (LLaVA-1.5 evaluation suite)

Official evaluation code for the LLaVA-1.5 part of the paper

> **Towards Robust Multimodal Large Language Models Against Jailbreak Attacks**
> Ziyi Yin, Yuanpu Cao, Han Liu, Ting Wang, Jinghui Chen, Fenglong Ma
> arXiv:[2502.00653](https://arxiv.org/abs/2502.00653) (2025).

This repo lets you reproduce all four evaluations reported in the paper for the LLaVA-1.5-7B
and LLaVA-1.5-13B SafeMLLM-trained models:

| #   | Benchmark        | Entry script                    | What it measures                                           |
| --- | ---------------- | ------------------------------- | ---------------------------------------------------------- |
| 1   | **ImgJP**        | `attacks/attack_llava_{7B,13B}.py` | White-box adversarial image jailbreak ASR on AdvBench    |
| 2   | **FigStep**      | `evals/eval_figstep.py`         | Black-box typographic-image jailbreak ASR (SafeBench-Tiny) |
| 3   | **MM-SafetyBench** | `evals/eval_mmbench.py`       | Black-box multimodal harmful-instruction ASR (Tiny split)  |
| 4   | **MM-Vet**       | `evals/eval_mmvet.py`           | General multimodal capability (utility) — defense sanity   |

If you also want the Qwen2.5-VL / InternVL-2.5 / MiniGPT variants, see the upstream
[Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model)
codebase — they will be added here in a follow-up release.

---

## Repo layout

```
SafeMLLM-LLaVA/
├── attacks/
│   ├── attack_llava_7B.py           # ImgJP attack on LLaVA-1.5-7B
│   └── attack_llava_13B.py          # ImgJP attack on LLaVA-1.5-13B
├── evals/
│   ├── eval_figstep.py              # FigStep / SafeBench-Tiny eval
│   ├── eval_mmbench.py              # MM-SafetyBench Tiny eval
│   └── eval_mmvet.py                # MM-Vet utility eval
├── llava/                           # LLaVA-1.5 model code (trimmed)
├── llava_llama_2/                   # LLaVA-LLaMA-2 model code (trimmed, used by torchattacks PGD)
├── llava_llama_2_utils/             # prompt_wrapper / generator helpers
├── torchattacks/                    # Slimmed torchattacks (only PGD-UAP for LLaVA)
├── dataset/
│   ├── advbench/harmful_behaviors.csv   # 520 AdvBench harmful behaviors (bundled)
│   ├── figstep/                     # ← DOWNLOAD MANUALLY (see dataset/README.md)
│   ├── mm-safety-bench/             # ← DOWNLOAD MANUALLY
│   └── mm-vet/                      # ← DOWNLOAD MANUALLY
├── scripts/
│   ├── run_L7B.sh                   # 4-in-1 evaluation pipeline for L7B
│   └── run_L13B.sh                  # 4-in-1 evaluation pipeline for L13B
├── environment.yml                  # conda env (Python 3.10 + torch 2.5.1 + transformers 4.31.0)
├── requirements.txt                 # pip-only equivalent
└── README.md                        # this file
```

---

## 1. Setup

### 1.1 Create the conda environment

```bash
conda env create -f environment.yml
conda activate safemllm-llava
```

If `conda env create` fails on your machine, fall back to plain pip in a fresh Python 3.10 venv:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

The environment is pinned to **Python 3.10 + PyTorch 2.5.1 + Transformers 4.31.0**, which is
exactly what produced the numbers in the paper. Newer transformers versions break the LLaVA-1.5
model code, so do **not** upgrade it.

### 1.2 Download model checkpoints

| Component                 | HuggingFace repo                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------- |
| LLaVA-1.5-7B base         | [`liuhaotian/llava-v1.5-7b`](https://huggingface.co/liuhaotian/llava-v1.5-7b)          |
| LLaVA-1.5-13B base        | [`liuhaotian/llava-v1.5-13b`](https://huggingface.co/liuhaotian/llava-v1.5-13b)        |
| SafeMLLM-LLaVA-7B (LoRA)  | [`ericyinyzy/SafeMLLM-LLaVA-7B`](https://huggingface.co/ericyinyzy/SafeMLLM-LLaVA-7B)  |
| SafeMLLM-LLaVA-13B (LoRA) | [`ericyinyzy/SafeMLLM-LLaVA-13B`](https://huggingface.co/ericyinyzy/SafeMLLM-LLaVA-13B)|

Quick download example:

```bash
pip install huggingface_hub
mkdir -p checkpoints
huggingface-cli download liuhaotian/llava-v1.5-7b        --local-dir checkpoints/llava-v1.5-7b
huggingface-cli download liuhaotian/llava-v1.5-13b       --local-dir checkpoints/llava-v1.5-13b
huggingface-cli download ericyinyzy/SafeMLLM-LLaVA-7B    --local-dir checkpoints/SafeMLLM-LLaVA-7B
huggingface-cli download ericyinyzy/SafeMLLM-LLaVA-13B   --local-dir checkpoints/SafeMLLM-LLaVA-13B
```

### 1.3 Download the evaluation datasets

`dataset/advbench/harmful_behaviors.csv` (used by ImgJP) is **already bundled**.

The three external benchmarks must be downloaded once. Each goes under a fixed sub-folder of
`dataset/`. See [dataset/README.md](dataset/README.md) for full instructions, or use the helper:

```bash
bash scripts/download_datasets.sh
```

Expected layout after download:

```
dataset/
├── advbench/harmful_behaviors.csv
├── figstep/
│   ├── SafeBench-Tiny.csv
│   └── SafeBench-Tiny/<query_ForbidQI_*.png>
├── mm-safety-bench/
│   ├── Tiny.json
│   └── Tiny/<image_id>.jpg
└── mm-vet/
    ├── mm-vet-transform.json
    └── images/<image_name>
```

---

## 2. Run the four evaluations

The two helper scripts encode all four commands. Just point them at your checkpoint paths:

```bash
# LLaVA-1.5-7B
export LLAVA7B_BASE=$PWD/checkpoints/llava-v1.5-7b
export SAFEMLLM_L7B=$PWD/checkpoints/SafeMLLM-LLaVA-7B
bash scripts/run_L7B.sh 0           # arg = GPU id

# LLaVA-1.5-13B
export LLAVA13B_BASE=$PWD/checkpoints/llava-v1.5-13b
export SAFEMLLM_L13B=$PWD/checkpoints/SafeMLLM-LLaVA-13B
bash scripts/run_L13B.sh 0
```

Or invoke the four commands individually (LLaVA-1.5-7B example shown):

```bash
# 1) ImgJP attack — produces advimg_L7B/L7B_safemllm.pt and result_L7B/test_L7B_safemllm.csv
python attacks/attack_llava_7B.py \
    --model-path  $SAFEMLLM_L7B \
    --model-base  $LLAVA7B_BASE \
    --advpath     advimg_L7B/L7B_safemllm.pt \
    --index 2 --iters 100

# 2) FigStep            — produces results_figstep/L7B_safemllm.csv
python evals/eval_figstep.py \
    --model-path $SAFEMLLM_L7B --model-base $LLAVA7B_BASE \
    --method 'L7B_safemllm'

# 3) MM-SafetyBench     — produces results_mmbench/L7B_safemllm.csv
python evals/eval_mmbench.py \
    --model-path $SAFEMLLM_L7B --model-base $LLAVA7B_BASE \
    --method 'L7B_safemllm'

# 4) MM-Vet utility     — produces results_utility_mmvet/L7B_safemllm.json
python evals/eval_mmvet.py \
    --model-path $SAFEMLLM_L7B --model-base $LLAVA7B_BASE \
    --outputpath results_utility_mmvet/L7B_safemllm.json
```

### Output locations

| Eval         | File                                                  |
| ------------ | ----------------------------------------------------- |
| ImgJP        | `result_L7B/test_<advname>.csv`, `advimg_L7B/*.pt`    |
| FigStep      | `results_figstep/<method>.csv`                        |
| MM-SafetyBench | `results_mmbench/<method>.csv`                      |
| MM-Vet       | `results_utility_mmvet/<method>.json`                 |

For MM-Vet, upload the produced JSON to the official
[MM-Vet evaluation server](https://github.com/yuweihao/MM-Vet) to obtain the final score.

---

## 3. Hardware

All four evaluations were run on a single NVIDIA A100-80G. Approximate VRAM usage:

| Model         | Inference | ImgJP attack (PGD, 100 iters) |
| ------------- | --------- | ----------------------------- |
| LLaVA-1.5-7B  | ~18 GB    | ~26 GB                        |
| LLaVA-1.5-13B | ~32 GB    | ~46 GB                        |

For 13B on a single 24 GB card, set `--iters 40` and pass `load_in_8bit=True` inside
`load_pretrained_model(...)`.

---

## 4. Cite

```bibtex
@article{yin2025safemllm,
  title   = {Towards Robust Multimodal Large Language Models Against Jailbreak Attacks},
  author  = {Yin, Ziyi and Cao, Yuanpu and Liu, Han and Wang, Ting and Chen, Jinghui and Ma, Fenglong},
  journal = {arXiv preprint arXiv:2502.00653},
  year    = {2025}
}
```

## 5. Acknowledgements

* Built upon [LLaVA](https://github.com/haotian-liu/LLaVA) and the upstream
  [Jailbreaking-Attack-against-MLLM](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) codebase.
* The PGD-UAP attack reuses [torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
  (slimmed; only `pgd_uap_llava.py` retained).
* External benchmarks: [FigStep](https://github.com/ThuCCSLab/FigStep),
  [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench),
  [MM-Vet](https://github.com/yuweihao/MM-Vet),
  [AdvBench](https://github.com/llm-attacks/llm-attacks).

## 6. License

Apache-2.0. See [LICENSE](LICENSE).
