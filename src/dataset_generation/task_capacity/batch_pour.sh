#!/usr/bin/env bash
#
# run_pour_compare.sh
#
# Runs batch_pour_compare.py N times, each with a different run_idx,
# picking a fresh pair each time. One simulator session per process.
#
# Usage:
#   chmod +x run_pour_compare.sh
#   ./run_pour_compare.sh
#
# Optional env var overrides:
#   CAPACITY_JSON=fillable_capacity.json NUM_RUNS=50 OUTPUT_ROOT=pour_results ./run_pour_compare.sh

CAPACITY_JSON="${CAPACITY_JSON:-fillable_capacity.json}"
NUM_RUNS="${NUM_RUNS:-300}"
OUTPUT_ROOT="${OUTPUT_ROOT:-batch_pour}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_pour.py"

echo "========================================================"
echo "  pour compare batch runner"
echo "  capacity_json : ${CAPACITY_JSON}"
echo "  num_runs      : ${NUM_RUNS}"
echo "  output_root   : ${OUTPUT_ROOT}"
echo "========================================================"

total=0
success=0
fail=0
skip=0

for (( run_idx=0; run_idx<NUM_RUNS; run_idx++ )); do

    out_dir="${OUTPUT_ROOT}/run_${run_idx}"

    # Skip if folder already exists
    if [ -d "${out_dir}" ]; then
        echo "[SKIP] run_${run_idx} — folder exists"
        (( skip++ ))
        continue
    fi

    (( total++ ))
    echo ""
    echo "----------------------------------------"
    echo "  run_idx=${run_idx}"
    echo "----------------------------------------"

    python "${SCRIPT}" \
        --capacity_json "${CAPACITY_JSON}" \
        --out_dir       "${out_dir}" \
        --run_idx       "${run_idx}"

    EXIT_CODE=$?
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        echo "[OK]    run_${run_idx}"
        (( success++ ))
    else
        echo "[ERROR] run_${run_idx} — exit code ${EXIT_CODE}"
        (( fail++ ))
    fi

done

echo ""
echo "========================================================"
echo "  Done.  total=${total}  success=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"