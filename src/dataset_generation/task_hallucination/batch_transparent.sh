#!/usr/bin/env bash
#
# run_transparent.sh
#
# Iterates over every (scene, room) pair in SCENES_DIR, calling
# batch_transparent.py once per run in a fresh Python process.
# One simulator session per process — no og.clear() needed.
#
# Usage:
#   chmod +x run_transparent.sh
#   ./run_transparent.sh
#
# Optional env var overrides:
#   SCENES_DIR=scenes5 RUNS_PER_ROOM=1 SCENE_FILTER=Merom ./run_transparent.sh

SCENES_DIR="${SCENES_DIR:-scenes5}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_transparent}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
SCENE_FILTER="${SCENE_FILTER:-}"   # optional substring filter on scene name
ROOM_FILTER="${ROOM_FILTER:-}"     # optional substring filter on room name

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/batch_transparent.py"

echo "========================================================"
echo "  transparent batch runner"
echo "  scenes_dir    : ${SCENES_DIR}"
echo "  room_objects  : ${ROOM_OBJECTS}"
echo "  runs_per_room : ${RUNS_PER_ROOM}"
echo "  output_root   : ${OUTPUT_ROOT}"
echo "  scene_filter  : '${SCENE_FILTER}'"
echo "  room_filter   : '${ROOM_FILTER}'"
echo "========================================================"

total=0
success=0
fail=0
skip=0

# ── Iterate scene JSON files ──────────────────────────────────────────────────
for scene_json in "${SCENES_DIR}"/*_scene_dict.json; do
    filename="$(basename "${scene_json}")"
    scene_name="${filename/_scene_dict.json/}"

    # Optional scene filter
    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    # ── Extract room names from scene JSON (keys of top-level object) ─────────
    mapfile -t rooms < <(python3 -c "
import json, sys
with open('${scene_json}') as f:
    d = json.load(f)
for r in d.keys():
    print(r)
")

    for room_name in "${rooms[@]}"; do
        # Optional room filter
        if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
            continue
        fi

        # ── Look up floor name from combined_room_object_list ─────────────────
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
            echo "[SKIP] ${scene_name} / ${room_name} — no floor found in room_objects"
            (( skip++ ))
            continue
        fi

        # ── One Python call per (scene, room, run_idx) ────────────────────────
        for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
            (( total++ ))
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
                --robot       "${ROBOT}" \
                --output_root "${OUTPUT_ROOT}"

            EXIT_CODE=$?
            if   [[ "${EXIT_CODE}" -eq 0 ]]; then
                echo "[OK]      scene=${scene_name} room=${room_name} run=${run_idx}"
                (( success++ ))
            elif [[ "${EXIT_CODE}" -eq 1 ]]; then
                echo "[PARTIAL] scene=${scene_name} room=${room_name} run=${run_idx} — placed but not visible"
                (( fail++ ))
            else
                echo "[ERROR]   scene=${scene_name} room=${room_name} run=${run_idx} — exit code ${EXIT_CODE}"
                (( fail++ ))
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "  Done.  total=${total}  success=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"