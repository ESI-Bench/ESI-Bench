#!/usr/bin/env bash
#
# run_slope.sh — iterates all (scene, room) pairs in scenes5, calls
# batch_slope.py once per run in a fresh Python process.
#
# Usage:
#   chmod +x run_slope.sh && ./run_slope.sh
#
# Overrides:
#   SCENES_DIR=scenes5 RUNS_PER_ROOM=3 SCENE_FILTER=Rs_int ./run_slope.sh

SCENES_DIR="${SCENES_DIR:-scenes5}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
OBJECTS_JSON="${OBJECTS_JSON:-slope_objects.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_slope}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_slope.py"

echo "========================================================"
echo "  run_slope.sh"
echo "  scenes_dir    : ${SCENES_DIR}"
echo "  runs_per_room : ${RUNS_PER_ROOM}"
echo "  output_root   : ${OUTPUT_ROOT}"
echo "  scene_filter  : '${SCENE_FILTER}'"
echo "  room_filter   : '${ROOM_FILTER}'"
echo "========================================================"

total=0; success=0; fail=0; skip=0

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
            echo "[SKIP] ${scene_name} / ${room_name} — no floor"
            (( skip++ )); continue
        fi

        for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
            (( total++ ))
            echo ""
            echo "----------------------------------------"
            echo "  scene=${scene_name}  room=${room_name}  run=${run_idx}"
            echo "  floor=${floor_name}"
            echo "----------------------------------------"

            python3 "${SCRIPT}" \
                --scene        "${scene_name}" \
                --room         "${room_name}" \
                --floor        "${floor_name}" \
                --run_idx      "${run_idx}" \
                --output_root  "${OUTPUT_ROOT}" \
                --objects_json "${OBJECTS_JSON}"

            EXIT_CODE=$?
            if [[ "${EXIT_CODE}" -eq 0 ]]; then
                echo "[OK]    scene=${scene_name} room=${room_name} run=${run_idx}"
                (( success++ ))
            else
                echo "[FAIL]  scene=${scene_name} room=${room_name} run=${run_idx} exit=${EXIT_CODE}"
                (( fail++ ))
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "  Done. total=${total} success=${success} fail=${fail} skip=${skip}"
echo "========================================================"