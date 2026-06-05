# Datasets

Only `advbench/harmful_behaviors.csv` is bundled in this repo. The three external
benchmarks must be downloaded once before running the corresponding evaluation.

## 1. AdvBench (bundled — used by ImgJP)

`advbench/harmful_behaviors.csv` ships with the repo. It is the standard 520-row
AdvBench harmful-behavior list from [llm-attacks](https://github.com/llm-attacks/llm-attacks).

## 2. FigStep / SafeBench-Tiny (used by `eval_figstep.py`)

Source repo: <https://github.com/ThuCCSLab/FigStep>

```bash
git clone https://github.com/ThuCCSLab/FigStep.git /tmp/FigStep
mkdir -p figstep
cp /tmp/FigStep/data/question/SafeBench-Tiny.csv figstep/SafeBench-Tiny.csv
cp -r /tmp/FigStep/data/images/SafeBench-Tiny    figstep/SafeBench-Tiny
```

Resulting layout:

```
dataset/figstep/
├── SafeBench-Tiny.csv          # 50 questions (category_id, task_id, question, ...)
└── SafeBench-Tiny/             # 50 *.png typographic-jailbreak images
```

## 3. MM-SafetyBench (used by `eval_mmbench.py`)

Source repo: <https://github.com/isXinLiu/MM-SafetyBench>

The "Tiny" subset used in our paper is a 50-prompt sample drawn from the full
MM-SafetyBench. Build it with:

```bash
mkdir -p mm-safety-bench/Tiny
# Place the Tiny.json file (one entry per prompt with key="0", key="1", ...) here:
cp /path/to/Tiny.json mm-safety-bench/Tiny.json
# And the corresponding 50 jpg images named "0.jpg", "1.jpg", ...:
cp /path/to/tiny_images/*.jpg mm-safety-bench/Tiny/
```

The expected JSON schema (per entry) is:

```json
{
  "0": { "question": "...", "id": 0 },
  "1": { "question": "...", "id": 1 },
  ...
}
```

If you don't already have `Tiny.json`, you can regenerate it from the upstream
MM-SafetyBench by sampling 50 prompts (deterministic seed used in the paper:
`numpy.random.seed(42)`).

## 4. MM-Vet (used by `eval_mmvet.py`)

Source: <https://github.com/yuweihao/MM-Vet>

```bash
mkdir -p mm-vet
# Download mm-vet.zip from the official release and unzip into mm-vet/
cd mm-vet && unzip /path/to/mm-vet.zip
# This script produces the flattened "transformed" JSON our evaluator expects:
python <<'PY'
import json, os
data = json.load(open('mm-vet.json'))
out = []
for qid, entry in data.items():
    out.append({
        "question_id": qid,
        "question":    entry["question"],
        "image":       entry["imagename"],
        "answer":      entry.get("answer", ""),
    })
with open('mm-vet-transform.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
PY
```

Final layout:

```
dataset/mm-vet/
├── mm-vet-transform.json       # list of {question_id, question, image, answer}
└── images/                     # all images referenced by the JSON
```

After running the evaluator, upload the produced
`results_utility_mmvet/<method>.json` to the
[MM-Vet evaluation server](https://github.com/yuweihao/MM-Vet) to obtain the score.
