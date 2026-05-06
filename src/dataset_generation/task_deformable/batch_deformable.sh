#!/usr/bin/env bash
#
# batch_deformable.sh
#
# For each run_idx, launch one full-scene OmniGibson process per scene and let
# the Python script traverse every pending room inside that loaded scene.
#
# Output layout:
#   OUTPUT_ROOT/<scene>/<room>/
#
# Optional env var overrides:
#   SCENES_DIR=/path/to/scenes
#   ROOM_OBJECTS=bddl3/bddl/generated_data/combined_room_object_list_future.json
#   ROBOT=R1
#   OUTPUT_ROOT=renders_cover_small_item
#   RUNS_PER_ROOM=5
#   PARALLEL_SCENES=1
#   SCENE_FILTER=Merom
#   ROOM_FILTER=bedroom
#   SKIP_RENDER=0
#   FAST_MODE=0
#   SCENE_WARMUP_STEPS=60
#   ITEM_ADD_STEPS=40
#   ITEM_SETTLE_STEPS=240
#   POST_ITEM_FREEZE_STEPS=10
#   CLOTH_ADD_STEPS=60
#   CLOTH_SETTLE_STEPS=300
#   CAPTURE_RENDER_STEPS=12
#   SMALL_ITEM_JSON=inference/small_portable_item_candidates_5to15cm.json
#   CLOTH_JSON=inference/cover_small_item_cloth_usable.json
#   QUESTION_COUNT=5
#   ROOM_TIMEOUT_SECONDS=1800

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_DIR="${SCENES_DIR:-${SCRIPT_DIR}/datasets/behavior-1k-assets/scenes}"
ROOM_OBJECTS="${ROOM_OBJECTS:-${SCRIPT_DIR}/bddl3/bddl/generated_data/combined_room_object_list_future.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/renders_deformable}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
PARALLEL_SCENES="${PARALLEL_SCENES:-1}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"
SKIP_RENDER="${SKIP_RENDER:-0}"
FAST_MODE="${FAST_MODE:-0}"
SCENE_WARMUP_STEPS="${SCENE_WARMUP_STEPS:-}"
ITEM_ADD_STEPS="${ITEM_ADD_STEPS:-}"
ITEM_SETTLE_STEPS="${ITEM_SETTLE_STEPS:-}"
POST_ITEM_FREEZE_STEPS="${POST_ITEM_FREEZE_STEPS:-}"
CLOTH_ADD_STEPS="${CLOTH_ADD_STEPS:-}"
CLOTH_SETTLE_STEPS="${CLOTH_SETTLE_STEPS:-}"
CAPTURE_RENDER_STEPS="${CAPTURE_RENDER_STEPS:-}"
SMALL_ITEM_JSON="${SMALL_ITEM_JSON:-${SCRIPT_DIR}/inference/small_portable_item_candidates_5to15cm.json}"
CLOTH_JSON="${CLOTH_JSON:-${SCRIPT_DIR}/inference/cover_small_item_cloth_usable.json}"
QUESTION_COUNT="${QUESTION_COUNT:-3}"
ROOM_TIMEOUT_SECONDS="${ROOM_TIMEOUT_SECONDS:-1800}"
MERGE_SCRIPT="${SCRIPT_DIR}/batch_deformable.py"
ATTEMPTED_ROOM_MARKER="cover_small_item_room_attempted.json"
SKIPPED_ROOM_MARKER="cover_small_item_room_skipped.json"
TASK_TYPE="cover_small_item_question_jsons/cover_small_item_cloth"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${SCRIPT_DIR}/${OUTPUT_ROOT}"
fi

should_skip_room_name() {
    local room_name="$1"
    [[ "${room_name}" == *garden* || "${room_name}" == *corridor* ]]
}

is_room_run_complete() {
    local scene_name="$1"
    local room_name="$2"
    local run_idx="$3"
    local run_root="${OUTPUT_ROOT}/${scene_name}/${room_name}"
    local attempted_marker="${run_root}/${ATTEMPTED_ROOM_MARKER}"
    local skipped_marker="${run_root}/${SKIPPED_ROOM_MARKER}"
    local last_q_idx=$(( QUESTION_COUNT - 1 ))
    local question_json="${run_root}/${TASK_TYPE}/q_$(printf '%03d' "${last_q_idx}").json"

    if [[ -f "${skipped_marker}" || -f "${question_json}" ]]; then
        return 0
    fi

    if [[ -f "${attempted_marker}" ]]; then
        python3 - "${attempted_marker}" <<'PY'
import json
import sys

marker_path = sys.argv[1]
try:
    with open(marker_path, encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    # If the marker is unreadable, retry the room instead of silently skipping it.
    sys.exit(1)

sys.exit(1 if data.get("status") == "error" else 0)
PY
        return $?
    fi

    return 1
}

echo "========================================================"
echo "  batch_deformable runner"
echo "  scenes_dir      : ${SCENES_DIR}"
echo "  room_objects    : ${ROOM_OBJECTS}"
echo "  robot           : ${ROBOT}"
echo "  output_root     : ${OUTPUT_ROOT}"
echo "  runs_per_room   : ${RUNS_PER_ROOM}"
echo "  parallel_scenes : ${PARALLEL_SCENES}"
echo "  scene_filter    : '${SCENE_FILTER}'"
echo "  room_filter     : '${ROOM_FILTER}'"
echo "  skip_render     : ${SKIP_RENDER}"
echo "  fast_mode       : ${FAST_MODE}"
echo "  small_item_json : ${SMALL_ITEM_JSON}"
echo "  cloth_json      : ${CLOTH_JSON}"
echo "  question_count  : ${QUESTION_COUNT}"
echo "  room_timeout_s  : ${ROOM_TIMEOUT_SECONDS}"
echo "  merge_script    : ${MERGE_SCRIPT##*/}"
echo "========================================================"

mkdir -p "${OUTPUT_ROOT}"

declare -a ALL_SCENES=()
declare -A SCENE_ROOMS_MAP=()
declare -A SCENE_FLOORS_MAP=()

for scene_dir in "${SCENES_DIR}"/*; do
    [[ -d "${scene_dir}" ]] || continue
    scene_name="${scene_dir##*/}"

    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    mapfile -t rooms < <(python3 - <<PY
import json
with open(${ROOM_OBJECTS@Q}) as f:
    data = json.load(f)
scenes = data.get("scenes", data)
for room_name in scenes.get(${scene_name@Q}, {}).keys():
    print(room_name)
PY
)

    [[ "${#rooms[@]}" -gt 0 ]] || continue

    scene_room_args=()
    scene_floor_args=()
    pending_count=0

    for room_name in "${rooms[@]}"; do
        if should_skip_room_name "${room_name}"; then
            continue
        fi
        if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
            continue
        fi

        floor_name="$(python3 - <<PY
import json, sys
with open(${ROOM_OBJECTS@Q}) as f:
    data = json.load(f)
scenes = data.get("scenes", data)
objs = scenes.get(${scene_name@Q}, {}).get(${room_name@Q}, [])
for obj_name in objs:
    if obj_name.startswith("floors-"):
        print(obj_name.replace("-", "_") + "_0")
        sys.exit(0)
PY
)"
        [[ -n "${floor_name}" ]] || continue

        scene_room_args+=("${room_name}")
        scene_floor_args+=("${floor_name}")
        (( pending_count += 1 ))
    done

    if (( pending_count == 0 )); then
        continue
    fi

    ALL_SCENES+=("${scene_name}")
    SCENE_ROOMS_MAP["${scene_name}"]="$(printf '%s\n' "${scene_room_args[@]}")"
    SCENE_FLOORS_MAP["${scene_name}"]="$(printf '%s\n' "${scene_floor_args[@]}")"
done

if [[ "${#ALL_SCENES[@]}" -eq 0 ]]; then
    echo "[DONE] No matching scenes / rooms found."
    exit 0
fi

write_room_timeout_markers() {
    local scene_name="$1"
    local room_name="$2"
    local floor_name="$3"
    local run_idx="$4"
    local exit_code="$5"
    local room_root="${OUTPUT_ROOT}/${scene_name}/${room_name}"
    local attempted_marker="${room_root}/${ATTEMPTED_ROOM_MARKER}"
    local skipped_marker="${room_root}/${SKIPPED_ROOM_MARKER}"
    local error_log_path="${room_root}/cover_small_item_errors.jsonl"

    mkdir -p "${room_root}"
    python3 - "${attempted_marker}" "${skipped_marker}" "${error_log_path}" "${scene_name}" "${room_name}" "${floor_name}" "${run_idx}" "${ROOM_TIMEOUT_SECONDS}" "${exit_code}" <<'PY'
import datetime as dt
import json
import sys

attempted_path, skipped_path, error_log_path, scene, room, floor, run_idx, timeout_s, exit_code = sys.argv[1:]
run_idx = int(run_idx)
timeout_s = int(timeout_s)
exit_code = int(exit_code)
payload = {
    "scene": scene,
    "room": room,
    "floor": floor,
    "run_idx": run_idx,
    "attempted": True,
    "status": "timeout",
    "question_count": 0,
    "skip_reason": f"room process exceeded timeout ({timeout_s}s) and was terminated",
    "error": f"TimeoutExpired: room process exceeded timeout ({timeout_s}s), exit_code={exit_code}",
    "error_log_path": error_log_path,
}
with open(attempted_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
with open(skipped_path, "w", encoding="utf-8") as f:
    json.dump({**payload, "skipped": True}, f, indent=2, ensure_ascii=False)
with open(error_log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "timestamp": dt.datetime.now().isoformat(),
        "event": "room_timeout",
        **payload,
    }, ensure_ascii=False) + "\n")
PY
}

run_room_job() {
    local scene_name="$1"
    local room_name="$2"
    local floor_name="$3"
    local run_idx="$4"
    local -a cmd=(
        python3 "${MERGE_SCRIPT}"
        --scene "${scene_name}"
        --room "${room_name}"
        --floor "${floor_name}"
        --robot "${ROBOT}"
        --run_idx "${run_idx}"
        --output_root "${OUTPUT_ROOT}"
        --small_item_json "${SMALL_ITEM_JSON}"
        --cloth_json "${CLOTH_JSON}"
        --question_count "${QUESTION_COUNT}"
        --exit_on_finish
    )

    if [[ "${SKIP_RENDER}" == "1" ]]; then
        cmd+=(--skip_render)
    fi
    if [[ "${FAST_MODE}" == "1" ]]; then
        cmd+=(--fast_mode)
    fi
    if [[ -n "${SCENE_WARMUP_STEPS}" ]]; then
        cmd+=(--scene_warmup_steps "${SCENE_WARMUP_STEPS}")
    fi
    if [[ -n "${ITEM_ADD_STEPS}" ]]; then
        cmd+=(--item_add_steps "${ITEM_ADD_STEPS}")
    fi
    if [[ -n "${ITEM_SETTLE_STEPS}" ]]; then
        cmd+=(--item_settle_steps "${ITEM_SETTLE_STEPS}")
    fi
    if [[ -n "${POST_ITEM_FREEZE_STEPS}" ]]; then
        cmd+=(--post_item_freeze_steps "${POST_ITEM_FREEZE_STEPS}")
    fi
    if [[ -n "${CLOTH_ADD_STEPS}" ]]; then
        cmd+=(--cloth_add_steps "${CLOTH_ADD_STEPS}")
    fi
    if [[ -n "${CLOTH_SETTLE_STEPS}" ]]; then
        cmd+=(--cloth_settle_steps "${CLOTH_SETTLE_STEPS}")
    fi
    if [[ -n "${CAPTURE_RENDER_STEPS}" ]]; then
        cmd+=(--capture_render_steps "${CAPTURE_RENDER_STEPS}")
    fi

    echo "[RUN] ${scene_name} / ${room_name} / run=${run_idx}"
    timeout --signal=TERM --kill-after=30s "${ROOM_TIMEOUT_SECONDS}" "${cmd[@]}"
    local exit_code=$?
    if [[ "${exit_code}" == "124" || "${exit_code}" == "137" ]]; then
        echo "[TIMEOUT] ${scene_name} / ${room_name} / run=${run_idx} exceeded ${ROOM_TIMEOUT_SECONDS}s, skipping room"
        write_room_timeout_markers "${scene_name}" "${room_name}" "${floor_name}" "${run_idx}" "${exit_code}"
        return 0
    fi
    return "${exit_code}"
}

declare -a ACTIVE_PIDS=()

wait_for_slot() {
    while (( ${#ACTIVE_PIDS[@]} >= PARALLEL_SCENES )); do
        local -a next_pids=()
        local pid
        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                next_pids+=("${pid}")
            else
                wait "${pid}" || true
            fi
        done
        ACTIVE_PIDS=("${next_pids[@]}")
        sleep 1
    done
}

wait_for_all_active() {
    local pid
    for pid in "${ACTIVE_PIDS[@]}"; do
        wait "${pid}"
    done
    ACTIVE_PIDS=()
}

for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
    echo "================ run_idx=${run_idx} ================"
    for scene_name in "${ALL_SCENES[@]}"; do
        mapfile -t scene_rooms <<< "${SCENE_ROOMS_MAP[$scene_name]}"
        mapfile -t scene_floors <<< "${SCENE_FLOORS_MAP[$scene_name]}"
        for (( idx=0; idx<${#scene_rooms[@]}; idx++ )); do
            room_name="${scene_rooms[$idx]}"
            floor_name="${scene_floors[$idx]}"
            if is_room_run_complete "${scene_name}" "${room_name}" "${run_idx}"; then
                echo "[SKIP ROOM] ${scene_name} / ${room_name} / run=${run_idx} - already complete"
                continue
            fi
            wait_for_slot
            run_room_job "${scene_name}" "${room_name}" "${floor_name}" "${run_idx}" &
            ACTIVE_PIDS+=("$!")
        done
    done
    wait_for_all_active
done

echo "[DONE] batch_deformable finished."
