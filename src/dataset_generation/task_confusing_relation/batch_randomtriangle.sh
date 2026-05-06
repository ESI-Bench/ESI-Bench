#!/usr/bin/env bash
#
# batch_randomtriangle.sh
#
# Iterates over every (scene, room) pair in SCENES_DIR, calling
# batch_randomtriangle.py once per run in a fresh Python process.
# One simulator session per process — no og.clear() needed.
#
# Loop order: run_idx is the OUTER loop, so run 0 completes across
# ALL rooms before run 1 begins (breadth-first over runs).
#
# Usage:
#   chmod +x batch_randomtriangle.sh
#   ./batch_randomtriangle.sh
#
# Optional env var overrides:
#   SCENES_DIR=scenes5 RUNS_PER_ROOM=1 SCENE_FILTER=Merom ./batch_randomtriangle.sh

SCENES_DIR="${SCENES_DIR:-scenes5}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-keys.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_randomtriangle}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-5}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_randomtriangle.py"

echo "========================================================"
echo "  batch_randomtriangle runner"
echo "  scenes_dir    : ${SCENES_DIR}"
echo "  room_objects  : ${ROOM_OBJECTS}"
echo "  runs_per_room : ${RUNS_PER_ROOM}"
echo "  output_root   : ${OUTPUT_ROOT}"
echo "  scene_filter  : '${SCENE_FILTER}'"
echo "  room_filter   : '${ROOM_FILTER}'"
echo "========================================================"

# ── Pre-scan: collect all (scene, room, floor) tuples ────────────────────────
declare -a SCENE_NAMES=()
declare -a ROOM_NAMES=()
declare -a FLOOR_NAMES=()

for scene_json in "${SCENES_DIR}"/*_scene_dict.json; do
    filename="$(basename "${scene_json}")"
    scene_name="${filename/_scene_dict.json/}"
    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    mapfile -t rooms < <(python3 -c "
import json
with open('${scene_json}') as f:
    d = json.load(f)
for r in d.keys():
    print(r)
")

    for room_name in "${rooms[@]}"; do
        if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
            continue
        fi

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
")"

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
echo "  rooms found   : ${n_rooms}"
echo "  total runs    : $(( n_rooms * RUNS_PER_ROOM ))"
echo "========================================================"

total=0
success=0
fail=0
skip=0

# ── Outer loop: run_idx — inner loop: rooms ───────────────────────────────────
for (( run_idx=1; run_idx<=RUNS_PER_ROOM; run_idx++ )); do
    echo ""
    echo "###################################################"
    echo "  Starting run_idx=${run_idx} across all ${n_rooms} rooms"
    echo "###################################################"

    for (( i=0; i<n_rooms; i++ )); do
        scene_name="${SCENE_NAMES[$i]}"
        room_name="${ROOM_NAMES[$i]}"
        floor_name="${FLOOR_NAMES[$i]}"

        (( total++ ))

        # ── Skip if output dir already exists ────────────────────────────────
        run_dir="${OUTPUT_ROOT}/${scene_name}/${room_name}_${run_idx}"
        if [[ -d "${run_dir}" ]]; then
            echo "[SKIP] ${scene_name} / ${room_name} / run=${run_idx} — ${run_dir} already exists"
            (( skip++ ))
            continue
        fi

        echo ""
        echo "----------------------------------------"
        echo "  scene=${scene_name}  room=${room_name}  run=${run_idx}"
        echo "  floor=${floor_name}"
        echo "----------------------------------------"

        python "${SCRIPT}" \
            --scene       "${scene_name}" \
            --room        "${room_name}" \
            --floor       "${floor_name}" \
            --run_idx     "${run_idx}" \
            --keys_json   "${KEYS_JSON}" \
            --robot       "${ROBOT}" \
            --output_root "${OUTPUT_ROOT}"

        EXIT_CODE=$?
        if   [[ "${EXIT_CODE}" -eq 0 ]]; then
            echo "[OK]      scene=${scene_name} room=${room_name} run=${run_idx}"
            (( success++ ))
        elif [[ "${EXIT_CODE}" -eq 1 ]]; then
            echo "[PARTIAL] scene=${scene_name} room=${room_name} run=${run_idx} — objects not visible"
            (( fail++ ))
        else
            echo "[ERROR]   scene=${scene_name} room=${room_name} run=${run_idx} — exit code ${EXIT_CODE}"
            (( fail++ ))
        fi
    done
done

echo ""
echo "========================================================"
echo "  Done.  total=${total}  success=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"