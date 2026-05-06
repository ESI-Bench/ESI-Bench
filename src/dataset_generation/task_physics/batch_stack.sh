#!/usr/bin/env bash
#
# batch_stacking.sh
#
# Iterates over every (scene, room) pair found in SCENES_DIR, calling
# batch_stacking.py once per (room, run_idx) in a fresh Python process.
# One OmniGibson session per process — no og.clear() issues between runs.
#
# Loop order: run_idx is the OUTER loop; rooms are the INNER loop.
# This means run 0 completes across ALL rooms before run 1 begins.
#
# Usage:
#   chmod +x batch_stacking.sh
#   ./batch_stacking.sh
#
# Optional env var overrides (set before calling the script):
#   SCENES_DIR=scenes5           Directory containing *_scene_dict.json files
#   ROOM_OBJECTS=bddl3/...json   combined_room_object_list_future.json path
#   KEYS_JSON=keys.json          Path to keys.json
#   ROBOT=R1                     Robot type
#   OUTPUT_ROOT=renders_stacking Output root directory
#   RUNS_PER_ROOM=5              How many run_idx values per room
#   N_OBJECTS=3                  2 or 3 objects in each stacking scene
#   SCENE_FILTER=Merom           If set, only process scenes matching this substring
#   ROOM_FILTER=                 If set, only process rooms matching this substring
#
# Exit codes from batch_stacking.py:
#   0 → at least one trial produced a stable stack
#   1 → no trial was stable (partial, metadata still saved)
#   2 → fatal error (scene/floor not found, etc.)

SCENES_DIR="${SCENES_DIR:-scenes5}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-keys.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_stacking}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
N_OBJECTS="${N_OBJECTS:-3}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_dependency.py"

echo "========================================================"
echo "  batch_stacking runner"
echo "  scenes_dir    : ${SCENES_DIR}"
echo "  room_objects  : ${ROOM_OBJECTS}"
echo "  keys_json     : ${KEYS_JSON}"
echo "  runs_per_room : ${RUNS_PER_ROOM}"
echo "  n_objects     : ${N_OBJECTS}"
echo "  output_root   : ${OUTPUT_ROOT}"
echo "  scene_filter  : '${SCENE_FILTER}'"
echo "  room_filter   : '${ROOM_FILTER}'"
echo "========================================================"

# ── Pre-scan: collect all valid (scene, room, floor) tuples ──────────────────
declare -a SCENE_NAMES=()
declare -a ROOM_NAMES=()
declare -a FLOOR_NAMES=()

for scene_json in "${SCENES_DIR}"/*_scene_dict.json; do
    [[ -f "${scene_json}" ]] || continue

    filename="${scene_json##*/}"
    scene_name="${filename/_scene_dict.json/}"

    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    # Extract room names from the scene dict JSON
    mapfile -t rooms < <(python3 -c "
import json, sys
with open('${scene_json}') as f:
    d = json.load(f)
for r in d.keys():
    print(r)
" 2>/dev/null)

    for room_name in "${rooms[@]}"; do
        if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
            continue
        fi

        # Look up floor object name for this (scene, room)
        floor_name="$(python3 -c "
import json, sys
with open('${ROOM_OBJECTS}') as f:
    data = json.load(f)
scenes = data.get('scenes', data)
objs = scenes.get('${scene_name}', {}).get('${room_name}', [])
for o in objs:
    if o.startswith('floors-'):
        print(o.replace('-', '_') + '_0')
        sys.exit(0)
" 2>/dev/null)"

        if [[ -z "${floor_name}" ]]; then
            echo "[SKIP] ${scene_name} / ${room_name} — no floor found"
            continue
        fi

        SCENE_NAMES+=("${scene_name}")
        ROOM_NAMES+=("${room_name}")
        FLOOR_NAMES+=("${floor_name}")
    done
done

n_rooms="${#SCENE_NAMES[@]}"
total_runs=$(( n_rooms * RUNS_PER_ROOM ))
echo "  rooms found   : ${n_rooms}"
echo "  total runs    : ${total_runs}"
echo "========================================================"

if [[ "${n_rooms}" -eq 0 ]]; then
    echo "[ERROR] No rooms found. Check SCENES_DIR and ROOM_OBJECTS paths."
    exit 1
fi

total=0
success=0
partial=0
fail=0
skip=0

# ── Outer: run_idx  ·  Inner: rooms ──────────────────────────────────────────
for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
    echo ""
    echo "###################################################"
    echo "  run_idx=${run_idx}  (across ${n_rooms} rooms)"
    echo "###################################################"

    for (( i=0; i<n_rooms; i++ )); do
        scene_name="${SCENE_NAMES[$i]}"
        room_name="${ROOM_NAMES[$i]}"
        floor_name="${FLOOR_NAMES[$i]}"

        (( total++ ))

        out_dir="${OUTPUT_ROOT}/${scene_name}/${room_name}/run_$(printf '%04d' ${run_idx})"
        meta_path="${out_dir}/metadata.json"

        # Skip if already completed
        if [[ -f "${meta_path}" ]]; then
            echo "[SKIP] already done: ${scene_name} / ${room_name} / run=${run_idx}"
            (( skip++ ))
            continue
        fi

        log_dir="${OUTPUT_ROOT}/logs/${scene_name}/${room_name}"
        mkdir -p "${log_dir}"
        log_file="${log_dir}/run_$(printf '%04d' ${run_idx}).log"

        echo ""
        echo "----------------------------------------------------"
        echo "  scene=${scene_name}"
        echo "  room =${room_name}"
        echo "  floor=${floor_name}"
        echo "  run  =${run_idx}"
        echo "  log  =${log_file}"
        echo "----------------------------------------------------"

        python3 "${SCRIPT}" \
            --scene       "${scene_name}" \
            --room        "${room_name}" \
            --floor       "${floor_name}" \
            --run_idx     "${run_idx}" \
            --keys_json   "${KEYS_JSON}" \
            --robot       "${ROBOT}" \
            --output_root "${OUTPUT_ROOT}" \
            --scenes_dir  "${SCENES_DIR}" \
            --n_objects   "${N_OBJECTS}" \
            2>&1 | tee "${log_file}"
        EXIT_CODE="${PIPESTATUS[0]}"

        if   [[ "${EXIT_CODE}" -eq 0 ]]; then
            echo "[OK]      ${scene_name} / ${room_name} / run=${run_idx} — stable stack found"
            (( success++ ))
        elif [[ "${EXIT_CODE}" -eq 1 ]]; then
            echo "[PARTIAL] ${scene_name} / ${room_name} / run=${run_idx} — no stable trial"
            (( partial++ ))
        else
            echo "[ERROR]   ${scene_name} / ${room_name} / run=${run_idx} — exit=${EXIT_CODE}"
            (( fail++ ))
        fi
    done
done

echo ""
echo "========================================================"
echo "  Done."
echo "  total=${total}  ok=${success}  partial=${partial}  fail=${fail}  skip=${skip}"
echo "========================================================"