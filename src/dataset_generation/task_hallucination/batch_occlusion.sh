#!/usr/bin/env bash
# Batch runner for batch_occlusion.py

SCENES_DIR="${SCENES_DIR:-scenes5}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-keys.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_occlusion}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"
PLACEMENT_MODE="${PLACEMENT_MODE:-sample}"
ASSET_MANIFEST="${ASSET_MANIFEST:-asset_manifest.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_occlusion.py"

echo "========================================================"
echo "  batch_occlusion runner"
echo "  scenes_dir      : ${SCENES_DIR}"
echo "  room_objects    : ${ROOM_OBJECTS}"
echo "  runs_per_room   : ${RUNS_PER_ROOM}"
echo "  output_root     : ${OUTPUT_ROOT}"
echo "  placement_mode  : ${PLACEMENT_MODE}"
echo "  asset_manifest : ${ASSET_MANIFEST}"
echo "  scene_filter    : '${SCENE_FILTER}'"
echo "  room_filter     : '${ROOM_FILTER}'"
echo "========================================================"

total=0
success=0
fail=0
skip=0

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

        for ((run_idx=0; run_idx<${RUNS_PER_ROOM}; run_idx++)); do
            ((total++))
            echo "[RUN] scene=${scene_name} room=${room_name} run_idx=${run_idx} floor=${floor_name}"
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
                echo "[OK] ${scene_name} / ${room_name} / ${run_idx}"
                ((success++))
            elif [[ $rc -eq 1 ]]; then
                echo "[PARTIAL] ${scene_name} / ${room_name} / ${run_idx}"
                ((fail++))
            else
                echo "[ERROR] ${scene_name} / ${room_name} / ${run_idx} (rc=${rc})"
                ((fail++))
            fi
        done
    done
done

echo "========================================================"
echo "Done. total=${total} success=${success} fail=${fail} skip=${skip}"
echo "========================================================"
