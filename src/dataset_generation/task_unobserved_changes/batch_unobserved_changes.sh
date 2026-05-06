#!/usr/bin/env bash
#
# batch_unobserved_changes.sh
#
# Iterates over every (scene, room) pair found in ROOM_OBJECTS and calls
# batch_unobserved_changes.py once per room.
#
# Usage:
#   chmod +x batch_unobserved_changes.sh
#   ./batch_unobserved_changes.sh
#
# Optional env var overrides:
#   SCENES_DIR=/path/to/scenes
#   ROOM_OBJECTS=bddl3/bddl/generated_data/combined_room_object_list_future.json
#   KEYS_JSON=keys.json
#   ROBOT=R1
#   OUTPUT_ROOT=renders_unobserved_changes
#   QUESTIONS_PER_TASK=2
#   PARALLEL_SCENES=2
#   SEED=7
#   SCENE_FILTER=Merom
#   ROOM_FILTER=bedroom
#   TASK_TYPES="change_detection change_identification current_state_reasoning"
#   SKIP_RENDER=1
#   LOAD_FULL_SCENE=1
#   DISABLE_RUNTIME_PHYSICS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_DIR="${SCENES_DIR:-/home/jliu/Desktop/project/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes}"
ROOM_OBJECTS="${ROOM_OBJECTS:-${SCRIPT_DIR}/bddl3/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-${SCRIPT_DIR}/keys.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_unobserved_changes}"
QUESTIONS_PER_TASK="${QUESTIONS_PER_TASK:-2}"
PARALLEL_SCENES="${PARALLEL_SCENES:-2}"
SEED="${SEED:-7}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"
TASK_TYPES="${TASK_TYPES:-change_detection change_identification current_state_reasoning}"
SKIP_RENDER="${SKIP_RENDER:-0}"
LOAD_FULL_SCENE="${LOAD_FULL_SCENE:-1}"
DISABLE_RUNTIME_PHYSICS="${DISABLE_RUNTIME_PHYSICS:-0}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${SCRIPT_DIR}/${OUTPUT_ROOT}"
fi

PY_SCRIPT="${SCRIPT_DIR}/batch_unobserved_changes.py"
read -r -a TASK_TYPES_ARR <<< "${TASK_TYPES}"

now_ts() {
    date +%s
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

format_duration() {
    local total_seconds="${1:-0}"
    local hours=$(( total_seconds / 3600 ))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$(( total_seconds % 60 ))
    printf '%02dh:%02dm:%02ds' "${hours}" "${minutes}" "${seconds}"
}

log_info() {
    echo "[$(timestamp)] $*"
}

log_stage_start() {
    local stage_name="$1"
    STAGE_START_TS["${stage_name}"]="$(now_ts)"
    log_info "[STAGE START] ${stage_name}"
}

log_stage_end() {
    local stage_name="$1"
    local end_ts
    local start_ts
    local elapsed
    end_ts="$(now_ts)"
    start_ts="${STAGE_START_TS["${stage_name}"]:-${end_ts}}"
    elapsed=$(( end_ts - start_ts ))
    STAGE_DURATION_SEC["${stage_name}"]="${elapsed}"
    log_info "[STAGE END] ${stage_name} | elapsed=$(format_duration "${elapsed}")"
}

should_skip_room_name() {
    local room_name="$1"
    [[ "${room_name}" == *garden* ]]
}

sanitize_path_component() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//[^[:alnum:]_.-]/_}"
    printf '%s\n' "${value}"
}

is_task_complete() {
    local room_root="$1"
    local task_type="$2"
    local output_task_dir="${room_root}/question_jsons/${task_type}"
    local json_count

    if [[ ! -d "${output_task_dir}" ]]; then
        return 1
    fi

    json_count=$(find "${output_task_dir}" -maxdepth 1 -name '*.json' | wc -l)
    (( json_count >= QUESTIONS_PER_TASK ))
}

is_room_complete() {
    local scene_name="$1"
    local room_name="$2"
    local room_root="${OUTPUT_ROOT}/${scene_name}/${room_name}"
    local task_type

    for task_type in "${TASK_TYPES_ARR[@]}"; do
        if ! is_task_complete "${room_root}" "${task_type}"; then
            return 1
        fi
    done

    return 0
}

echo "========================================================"
echo "  batch_unobserved_changes runner"
echo "  scenes_dir             : ${SCENES_DIR}"
echo "  room_objects           : ${ROOM_OBJECTS}"
echo "  keys_json              : ${KEYS_JSON}"
echo "  questions_per_task     : ${QUESTIONS_PER_TASK}"
echo "  parallel_scenes        : ${PARALLEL_SCENES}"
echo "  seed                   : ${SEED}"
echo "  output_root            : ${OUTPUT_ROOT}"
echo "  skip_render            : ${SKIP_RENDER}"
echo "  load_full_scene        : ${LOAD_FULL_SCENE}"
echo "  disable_runtime_physics: ${DISABLE_RUNTIME_PHYSICS}"
echo "  scene_filter           : '${SCENE_FILTER}'"
echo "  room_filter            : '${ROOM_FILTER}'"
echo "  script                 : ${PY_SCRIPT##*/}"
echo "  task_types             :"
for task_type in "${TASK_TYPES_ARR[@]}"; do
    echo "    - ${task_type}"
done
echo "========================================================"

declare -a UNIQUE_SCENES=()
declare -A SCENE_SEEN=()
declare -A ROOMS_BY_SCENE=()
declare -A FLOOR_BY_SCENE_ROOM=()
declare -A SCENE_JOB_ROOMS=()
declare -A SCENE_JOB_FLOORS=()
declare -A STAGE_START_TS=()
declare -A STAGE_DURATION_SEC=()

log_stage_start "load_room_metadata"

while IFS=$'\t' read -r parsed_scene_name parsed_room_name parsed_floor_name; do
    [[ -n "${parsed_scene_name}" && -n "${parsed_room_name}" ]] || continue
    if [[ -n "${ROOMS_BY_SCENE["${parsed_scene_name}"]:-}" ]]; then
        ROOMS_BY_SCENE["${parsed_scene_name}"]+=$'\n'"${parsed_room_name}"
    else
        ROOMS_BY_SCENE["${parsed_scene_name}"]="${parsed_room_name}"
    fi
    FLOOR_BY_SCENE_ROOM["${parsed_scene_name}"$'\t'"${parsed_room_name}"]="${parsed_floor_name}"
done < <(
    python3 - "${ROOM_OBJECTS}" <<'PY'
import json
import sys

room_objects_path = sys.argv[1]
with open(room_objects_path, encoding="utf-8") as f:
    data = json.load(f)

scenes = data.get("scenes", data)
for scene_name, rooms in scenes.items():
    if not isinstance(rooms, dict):
        continue
    for room_name, objs in rooms.items():
        floor_name = ""
        if isinstance(objs, dict):
            obj_names = objs.keys()
        elif isinstance(objs, list):
            obj_names = objs
        else:
            obj_names = []
        for obj_name in obj_names:
            if isinstance(obj_name, str) and obj_name.startswith("floors-"):
                floor_name = obj_name.replace("-", "_") + "_0"
                break
        print(f"{scene_name}\t{room_name}\t{floor_name}")
PY
)

log_stage_end "load_room_metadata"

log_stage_start "discover_pending_jobs"

for scene_dir in "${SCENES_DIR}"/*; do
    [[ -d "${scene_dir}" ]] || continue
    scene_name="${scene_dir##*/}"

    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    if [[ -z "${ROOMS_BY_SCENE["${scene_name}"]:-}" ]]; then
        continue
    fi

    mapfile -t rooms <<< "${ROOMS_BY_SCENE["${scene_name}"]}"
    pending_rooms=()
    pending_floors=()

    for room_name in "${rooms[@]}"; do
        if should_skip_room_name "${room_name}"; then
            echo "[SKIP ROOM] ${scene_name} / ${room_name} - room name contains garden"
            continue
        fi
        if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
            continue
        fi
        if is_room_complete "${scene_name}" "${room_name}"; then
            echo "[SKIP ROOM] ${scene_name} / ${room_name} - existing outputs detected"
            continue
        fi

        floor_name="${FLOOR_BY_SCENE_ROOM["${scene_name}"$'\t'"${room_name}"]:-}"
        if [[ -z "${floor_name}" ]]; then
            echo "[SKIP ROOM] ${scene_name} / ${room_name} - no floor found"
            continue
        fi

        pending_rooms+=("${room_name}")
        pending_floors+=("${floor_name}")
    done

    if (( ${#pending_rooms[@]} == 0 )); then
        continue
    fi

    if [[ -z "${SCENE_SEEN["${scene_name}"]:-}" ]]; then
        UNIQUE_SCENES+=("${scene_name}")
        SCENE_SEEN["${scene_name}"]=1
    fi
    printf -v SCENE_JOB_ROOMS["${scene_name}"] '%s\n' "${pending_rooms[@]}"
    printf -v SCENE_JOB_FLOORS["${scene_name}"] '%s\n' "${pending_floors[@]}"
done

log_stage_end "discover_pending_jobs"

n_rooms=0
for scene_name in "${UNIQUE_SCENES[@]}"; do
    mapfile -t scene_rooms <<< "${SCENE_JOB_ROOMS["${scene_name}"]}"
    n_rooms=$(( n_rooms + ${#scene_rooms[@]} ))
done

num_tasks="${#TASK_TYPES_ARR[@]}"
total_task_outputs=$(( n_rooms * num_tasks ))
echo "  rooms found            : ${n_rooms}"
echo "  scene jobs             : ${#UNIQUE_SCENES[@]}"
echo "  task outputs           : ${total_task_outputs}"
echo "========================================================"

if [[ "${n_rooms}" -eq 0 ]]; then
    echo "[DONE] No pending rooms found."
    exit 0
fi

declare -a ACTIVE_PIDS=()
declare -A PID_TO_SCENE=()
declare -A PID_TO_ROOM=()
declare -A PID_TO_LOG=()
declare -A PID_TO_START_TS=()
declare -A ROOM_ELAPSED_SEC=()

total=0
success=0
fail=0

reap_jobs() {
    local wait_for_one="${1:-0}"

    while (( ${#ACTIVE_PIDS[@]} > 0 )); do
        local completed_any=0
        local -a remaining_pids=()
        local pid
        local exit_code

        for pid in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                remaining_pids+=("${pid}")
                continue
            fi

            if wait "${pid}"; then
                exit_code=0
            else
                exit_code=$?
            fi

            scene_name="${PID_TO_SCENE[$pid]}"
            room_name="${PID_TO_ROOM[$pid]}"
            log_file="${PID_TO_LOG[$pid]}"
            room_key="${scene_name}/${room_name}"
            end_ts="$(now_ts)"
            start_ts="${PID_TO_START_TS[$pid]:-${end_ts}}"
            elapsed=$(( end_ts - start_ts ))
            ROOM_ELAPSED_SEC["${room_key}"]="${elapsed}"
            room_root="${OUTPUT_ROOT}/${scene_name}/${room_name}"
            room_ok=1
            for task_type in "${TASK_TYPES_ARR[@]}"; do
                if is_task_complete "${room_root}" "${task_type}"; then
                    log_info "[OK] ${scene_name} / ${room_name} / ${task_type} | room_elapsed=$(format_duration "${elapsed}")"
                    (( success++ ))
                else
                    log_info "[ERROR] ${scene_name} / ${room_name} / ${task_type} - see ${log_file} (exit=${exit_code}) | room_elapsed=$(format_duration "${elapsed}")"
                    (( fail++ ))
                    room_ok=0
                fi
            done
            if [[ "${room_ok}" -eq 1 ]]; then
                log_info "[ROOM DONE] ${scene_name} / ${room_name} | elapsed=$(format_duration "${elapsed}") | log=${log_file}"
            fi

            unset PID_TO_SCENE["$pid"] PID_TO_ROOM["$pid"] PID_TO_LOG["$pid"] PID_TO_START_TS["$pid"]
            completed_any=1
        done

        ACTIVE_PIDS=("${remaining_pids[@]}")
        if (( completed_any == 1 && wait_for_one == 1 )); then
            break
        fi
        if (( completed_any == 0 )); then
            sleep 1
        fi
        if (( wait_for_one == 0 && ${#ACTIVE_PIDS[@]} == 0 )); then
            break
        fi
    done
}

log_stage_start "run_room_jobs"

for scene_name in "${UNIQUE_SCENES[@]}"; do
    mapfile -t scene_rooms <<< "${SCENE_JOB_ROOMS["${scene_name}"]}"
    mapfile -t scene_floors <<< "${SCENE_JOB_FLOORS["${scene_name}"]}"
    (( ${#scene_rooms[@]} > 0 )) || continue

    total=$(( total + ${#scene_rooms[@]} * num_tasks ))
    log_dir="${OUTPUT_ROOT}/logs/${scene_name}"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/${scene_name}_unobserved_changes.log"

    echo "----------------------------------------------------"
    echo "  scene=${scene_name}"
    echo "  rooms=${#scene_rooms[@]}"
    echo "  seed =${SEED}"
    echo "  root =${OUTPUT_ROOT}/${scene_name}"
    echo "  log  =${log_file}"
    echo "----------------------------------------------------"

    cmd=(
        python3 "${PY_SCRIPT}"
        --scene "${scene_name}"
        --seed "${SEED}"
        --keys_json "${KEYS_JSON}"
        --robot "${ROBOT}"
        --output_root "${OUTPUT_ROOT}"
        --questions_per_task "${QUESTIONS_PER_TASK}"
        --task_types "${TASK_TYPES_ARR[@]}"
    )

    if [[ "${SKIP_RENDER}" == "1" ]]; then
        cmd+=(--skip_render)
    fi
    if [[ "${LOAD_FULL_SCENE}" == "1" ]]; then
        cmd+=(--load_full_scene)
    fi
    if [[ "${DISABLE_RUNTIME_PHYSICS}" == "1" ]]; then
        cmd+=(--disable_runtime_physics)
    fi

    # Run one room per process because batch_unobserved_changes.py currently
    # accepts a single --room / --floor pair.
    for idx in "${!scene_rooms[@]}"; do
        room_name="${scene_rooms[$idx]}"
        floor_name="${scene_floors[$idx]}"
        room_cmd=("${cmd[@]}" --room "${room_name}" --floor "${floor_name}")
        room_log="${log_dir}/$(sanitize_path_component "${room_name}").log"
        log_info "[ROOM START] ${scene_name} / ${room_name} | floor=${floor_name} | log=${room_log}"
        "${room_cmd[@]}" > "${room_log}" 2>&1 &
        pid=$!
        ACTIVE_PIDS+=("${pid}")
        PID_TO_SCENE["${pid}"]="${scene_name}"
        PID_TO_ROOM["${pid}"]="${room_name}"
        PID_TO_LOG["${pid}"]="${room_log}"
        PID_TO_START_TS["${pid}"]="$(now_ts)"

        while (( ${#ACTIVE_PIDS[@]} >= PARALLEL_SCENES )); do
            reap_jobs 1
        done
    done
done

reap_jobs 0

log_stage_end "run_room_jobs"

echo "========================================================"
echo "  Done."
echo "  total=${total}  ok=${success}  fail=${fail}"
echo "  stage timings:"
for stage_name in load_room_metadata discover_pending_jobs run_room_jobs; do
    echo "    - ${stage_name}: $(format_duration "${STAGE_DURATION_SEC["${stage_name}"]:-0}")"
done
if (( ${#ROOM_ELAPSED_SEC[@]} > 0 )); then
    echo "  slowest rooms:"
    while IFS=$'\t' read -r elapsed room_key; do
        [[ -n "${room_key}" ]] || continue
        echo "    - ${room_key}: $(format_duration "${elapsed}")"
    done < <(
        for room_key in "${!ROOM_ELAPSED_SEC[@]}"; do
            printf '%s\t%s\n' "${ROOM_ELAPSED_SEC["${room_key}"]}" "${room_key}"
        done | sort -rn | head -n 10
    )
fi
echo "========================================================"
