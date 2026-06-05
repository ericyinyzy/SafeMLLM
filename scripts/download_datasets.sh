#!/usr/bin/env bash
# One-shot downloader for the FigStep / MM-SafetyBench / MM-Vet eval datasets.
# AdvBench is bundled in the repo and does NOT need to be downloaded.
#
# Usage:  bash scripts/download_datasets.sh
#
# This script clones each upstream repo into /tmp and copies the relevant files into
# dataset/{figstep,mm-safety-bench,mm-vet}/.   Re-runnable; it skips already-present files.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p dataset/figstep dataset/mm-safety-bench dataset/mm-vet

# ---------- FigStep / SafeBench-Tiny ----------
if [[ ! -f dataset/figstep/SafeBench-Tiny.csv ]]; then
    echo "[1/3] Cloning FigStep..."
    rm -rf /tmp/_figstep && git clone --depth 1 https://github.com/ThuCCSLab/FigStep.git /tmp/_figstep
    cp /tmp/_figstep/data/question/SafeBench-Tiny.csv dataset/figstep/SafeBench-Tiny.csv
    cp -r /tmp/_figstep/data/images/SafeBench-Tiny    dataset/figstep/
fi

# ---------- MM-SafetyBench ----------
if [[ ! -f dataset/mm-safety-bench/Tiny.json ]]; then
    cat <<'MSG'
[2/3] MM-SafetyBench:
    The "Tiny" 50-prompt subset must be supplied manually -- the upstream repo
    publishes the full bench, not the Tiny split we used in the paper. Please copy
        Tiny.json   -> dataset/mm-safety-bench/Tiny.json
        Tiny/*.jpg  -> dataset/mm-safety-bench/Tiny/
    See dataset/README.md for the expected JSON schema.
MSG
fi

# ---------- MM-Vet ----------
if [[ ! -f dataset/mm-vet/mm-vet-transform.json ]]; then
    cat <<'MSG'
[3/3] MM-Vet:
    Please download mm-vet.zip from https://github.com/yuweihao/MM-Vet/releases
    into dataset/mm-vet/ and run the JSON-transformation snippet from
    dataset/README.md.
MSG
fi

echo "Done. AdvBench is already bundled at dataset/advbench/harmful_behaviors.csv."
