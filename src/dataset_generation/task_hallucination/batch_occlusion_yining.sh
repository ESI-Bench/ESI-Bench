#!/usr/bin/env bash
# Batch runner for batch_occlusion.py
# Outer loop: run_idx (0..RUNS_PER_ROOM-1)
# Inner loop: all scenes / rooms
# This ensures run 0 is done for every room before run 1 starts, etc.

SCENES_DIR="${SCENES_DIR:-scenes5}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-keys.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_occlusion_v2}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-5}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"
PLACEMENT_MODE="${PLACEMENT_MODE:-sample}"
ASSET_MANIFEST="${ASSET_MANIFEST:-asset_manifest.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_occlusion_yining.py"

echo "========================================================"
echo "  batch_occlusion runner"
echo "  scenes_dir      : ${SCENES_DIR}"
echo "  room_objects    : ${ROOM_OBJECTS}"
echo "  runs_per_room   : ${RUNS_PER_ROOM}"
echo "  output_root     : ${OUTPUT_ROOT}"
echo "  placement_mode  : ${PLACEMENT_MODE}"
echo "  asset_manifest  : ${ASSET_MANIFEST}"
echo "  scene_filter    : '${SCENE_FILTER}'"
echo "  room_filter     : '${ROOM_FILTER}'"
echo "  loop order      : run_idx outer, rooms inner"
echo "========================================================"

total=0
success=0
fail=0
skip=0

# Build list of (scene, room, floor) triples upfront
declare -a SCENE_LIST
declare -a ROOM_LIST
declare -a FLOOR_LIST

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
            ((skip++))
            continue
        fi

        SCENE_LIST+=("${scene_name}")
        ROOM_LIST+=("${room_name}")
        FLOOR_LIST+=("${floor_name}")
    done
done

num_rooms="${#SCENE_LIST[@]}"
echo "[info] Found ${num_rooms} valid rooms. Running ${RUNS_PER_ROOM} runs each."
echo "========================================================"

# Outer loop: run index
for ((run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++)); do
    echo ""
    echo "========================================================"
    echo "  Starting run_idx=${run_idx} across all ${num_rooms} rooms"
    echo "========================================================"

    # Inner loop: rooms
    for ((i=0; i<num_rooms; i++)); do
        scene_name="${SCENE_LIST[$i]}"
        room_name="${ROOM_LIST[$i]}"
        floor_name="${FLOOR_LIST[$i]}"

        ((total++))
        echo "[RUN] run_idx=${run_idx} scene=${scene_name} room=${room_name} floor=${floor_name}"

        python3 "${SCRIPT}" \
            --scene "${scene_name}" \
            --room "${room_name}" \
            --floor "${floor_name}" \
            --run_idx "${run_idx}" \
            --keys_json "${KEYS_JSON}" \
            --robot "${ROBOT}" \
            --output_root "${OUTPUT_ROOT}" \
            --placement_mode "${PLACEMENT_MODE}" \
            --asset_manifest "${ASSET_MANIFEST}"

        rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "[OK]      run_idx=${run_idx} ${scene_name} / ${room_name}"
            ((success++))
        elif [[ $rc -eq 1 ]]; then
            echo "[PARTIAL] run_idx=${run_idx} ${scene_name} / ${room_name}"
            ((fail++))
        else
            echo "[ERROR]   run_idx=${run_idx} ${scene_name} / ${room_name} (rc=${rc})"
            ((fail++))
        fi
    done

    echo "[run_idx=${run_idx} done] success=${success} fail=${fail} so far"
done

echo ""
echo "========================================================"
echo "Done. total=${total} success=${success} fail=${fail} skip=${skip}"
echo "========================================================"