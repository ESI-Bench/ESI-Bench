#!/usr/bin/env bash
#
# run_water_fill.sh
#
# Iterates over every (category, model) pair in fillable_manifest.json,
# calling batch_water_fill.py once per instance in a fresh Python process.
# One simulator session per process — no og.clear() needed.
#
# Usage:
#   chmod +x run_water_fill.sh
#   ./run_water_fill.sh
#
# Optional env var overrides:
#   MANIFEST=fillable_manifest.json OUTPUT_ROOT=water_fill_results CATEGORY_FILTER=bowl ./run_water_fill.sh

MANIFEST="${MANIFEST:-fillable_manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-water_fill_results}"
CATEGORY_FILTER="${CATEGORY_FILTER:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_water.py"
MANIFEST_SCRIPT="${SCRIPT_DIR}/build_fillable_manifest.py"

echo "========================================================"
echo "  water fill batch runner"
echo "  manifest        : ${MANIFEST}"
echo "  output_root     : ${OUTPUT_ROOT}"
echo "  category_filter : '${CATEGORY_FILTER}'"
echo "========================================================"

# ── Build manifest if it doesn't exist yet ────────────────────────────────────
if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo "[manifest] ${MANIFEST} not found — building now..."
    python "${MANIFEST_SCRIPT}"
else
    echo "[manifest] ${MANIFEST} already exists — skipping rebuild."
    echo "           Delete it to force a rescan."
fi

total=0
success=0
fail=0
skip=0

# ── Iterate (category, model) pairs from manifest ────────────────────────────
while IFS=' ' read -r category model; do

    # Optional category filter
    if [[ -n "${CATEGORY_FILTER}" && "${category}" != *"${CATEGORY_FILTER}"* ]]; then
        continue
    fi

    # Skip if output folder already exists
    out_dir="${OUTPUT_ROOT}/${category}_${model}"
    if [ -d "${out_dir}" ]; then
        echo "[SKIP] ${category} / ${model} — folder exists"
        (( skip++ ))
        continue
    fi

    (( total++ ))
    echo ""
    echo "----------------------------------------"
    echo "  category=${category}  model=${model}"
    echo "----------------------------------------"

    python "${SCRIPT}" \
        --category "${category}" \
        --model    "${model}" \
        --out_dir  "${out_dir}"

    EXIT_CODE=$?
    if   [[ "${EXIT_CODE}" -eq 0 ]]; then
        echo "[OK]      category=${category} model=${model}"
        (( success++ ))
    elif [[ "${EXIT_CODE}" -eq 1 ]]; then
        echo "[EMPTY]   category=${category} model=${model} — no particles captured"
        (( fail++ ))
    else
        echo "[ERROR]   category=${category} model=${model} — exit code ${EXIT_CODE}"
        (( fail++ ))
    fi

done < <(python3 -c "
import json, random
with open('${MANIFEST}') as f:
    m = json.load(f)
pairs = [(cat, mdl) for cat, models in m.items() for mdl in models]
random.shuffle(pairs)
for cat, mdl in pairs:
    print(cat, mdl)
")

echo ""
echo "========================================================"
echo "  Done.  total=${total}  success=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"