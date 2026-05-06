#!/usr/bin/env bash
#
# batch_counting_merge.sh
#
# Iterates over every (scene, room) pair found in SCENES_DIR, calling
# batch_counting_merge.py once per room. The Python script generates all
# counting task types in a single OmniGibson session.
#
# Usage:
#   chmod +x batch_counting_merge.sh
#   ./batch_counting_merge.sh
#
# Optional env var overrides:
#   SCENES_DIR=/path/to/scenes
#   ROOM_OBJECTS=bddl3/bddl/generated_data/combined_room_object_list_future.json
#   KEYS_JSON=keys.json
#   KEYS_CLIP_TOP3_JSON=keys_clip_top3.json
#   ROBOT=R1
#   OUTPUT_ROOT=renders_counting
#   MAX_PER_CASE=5
#   PARALLEL_SCENES=2
#   SEED=7
#   SCENE_FILTER=Merom
#   ROOM_FILTER=bedroom
#   TASK_TYPES="hidden_in_box"
#   SKIP_RENDER=1
#   LOAD_FULL_SCENE=1
#   DISABLE_RUNTIME_PHYSICS=1
#   CACHE_CLEANUP_INTERVAL=1

OMNIGIBSON_APPDATA_PATH=/data/$USER/omnigibson_appdata
CACHE_CLEANUP_INTERVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_DIR="${SCENES_DIR:-/home/jliu/Desktop/project/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes}"
ROOM_OBJECTS="${ROOM_OBJECTS:-${SCRIPT_DIR}/bddl3/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-${SCRIPT_DIR}/keys.json}"
KEYS_CLIP_TOP3_JSON="${KEYS_CLIP_TOP3_JSON:-${SCRIPT_DIR}/keys_clip_top3.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_counting_new}"
MAX_PER_CASE="${MAX_PER_CASE:-5}"
PARALLEL_SCENES="${PARALLEL_SCENES:-2}"
SEED="${SEED:-7}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"
TASK_TYPES="${TASK_TYPES:-hidden_by_others observation_divided semantic_fault hidden_in_box observation_merged light_change}"
SKIP_RENDER="${SKIP_RENDER:-0}"
LOAD_FULL_SCENE="${LOAD_FULL_SCENE:-1}"
DISABLE_RUNTIME_PHYSICS="${DISABLE_RUNTIME_PHYSICS:-1}"

if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${SCRIPT_DIR}/${OUTPUT_ROOT}"
fi

DEFAULT_OG_APPDATA_PATH="${SCRIPT_DIR}/OmniGibson/appdata"
if [[ -z "${OMNIGIBSON_APPDATA_PATH:-}" && -d /data ]]; then
    OMNIGIBSON_APPDATA_PATH="/data/${USER}/omnigibson_appdata"
fi
OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-${DEFAULT_OG_APPDATA_PATH}}"
CACHE_CLEANUP_INTERVAL="${CACHE_CLEANUP_INTERVAL:-0}"

MERGE_SCRIPT="${SCRIPT_DIR}/batch_counting_merge.py"
ATTEMPTED_ROOM_MARKER="counting_room_attempted.json"
read -r -a TASK_TYPES_ARR <<< "${TASK_TYPES}"

cleanup_omnigibson_cache() {
    local appdata_root="${1:-${OMNIGIBSON_APPDATA_PATH}}"
    local cache_dir="${appdata_root}/global/cache"

    if [[ ! -d "${cache_dir}" ]]; then
        return 0
    fi

    echo "[CACHE] clearing ${cache_dir}"
    find "${cache_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
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

PROGRESS_BAR_WIDTH="${PROGRESS_BAR_WIDTH:-32}"
PROGRESS_IS_TTY=0
if [[ -t 1 ]]; then
    PROGRESS_IS_TTY=1
fi

clear_progress_line() {
    if (( PROGRESS_IS_TTY == 1 )); then
        printf '\r\033[2K'
    fi
}

render_scene_progress() {
    local completed="${1:-0}"
    local total_jobs="${2:-0}"
    local active_jobs="${3:-0}"
    local phase="${4:-running}"
    local last_scene="${5:-}"

    if (( total_jobs <= 0 )); then
        return 0
    fi

    local filled=0
    if (( total_jobs > 0 )); then
        filled=$(( completed * PROGRESS_BAR_WIDTH / total_jobs ))
    fi
    if (( filled > PROGRESS_BAR_WIDTH )); then
        filled="${PROGRESS_BAR_WIDTH}"
    fi
    local empty=$(( PROGRESS_BAR_WIDTH - filled ))
    local bar
    printf -v bar '%*s' "${filled}" ''
    bar="${bar// /#}"
    local pad
    printf -v pad '%*s' "${empty}" ''
    pad="${pad// /-}"

    local percent=$(( completed * 100 / total_jobs ))
    local message="[SCENE PROGRESS] [${bar}${pad}] ${completed}/${total_jobs} (${percent}%%) active=${active_jobs} phase=${phase}"
    if [[ -n "${last_scene}" ]]; then
        message="${message} scene=${last_scene}"
    fi

    if (( PROGRESS_IS_TTY == 1 )); then
        printf '\r\033[2K%s' "${message}"
        if [[ "${phase}" == "done" ]]; then
            printf '\n'
        fi
    else
        printf '%s\n' "${message}"
    fi
}

is_task_complete() {
    local room_root="$1"
    local task_type="$2"
    local output_task_dir="${room_root}/counting_question_jsons/${task_type}"
    local required_count="${MAX_PER_CASE}"
    local json_count

    if [[ ! -d "${output_task_dir}" ]]; then
        return 1
    fi

    json_count=$(find "${output_task_dir}" -maxdepth 1 -name '*.json' | wc -l)
    (( json_count >= required_count ))
}

is_room_attempted() {
    local room_root="$1"

    [[ -f "${room_root}/${ATTEMPTED_ROOM_MARKER}" ]]
}

is_room_nonempty() {
    local room_root="$1"

    [[ -d "${room_root}" ]] || return 1
    find "${room_root}" -mindepth 1 -print -quit | grep -q .
}

is_room_complete() {
    local scene_name="$1"
    local room_name="$2"
    local room_root="${OUTPUT_ROOT}/${scene_name}/${room_name}"
    local task_type

    if is_room_attempted "${room_root}"; then
        return 0
    fi

    # Only treat the room as complete once every task has produced the
    # requested number of questions, so reruns can keep filling partial outputs.
    for task_type in "${TASK_TYPES_ARR[@]}"; do
        if ! is_task_complete "${room_root}" "${task_type}"; then
            return 1
        fi
    done

    return 0
}

scene_processed_counts() {
    local scene_name="$1"
    shift
    local room_names=("$@")

    local room_name
    local eligible_room_count=0
    local processed_room_count=0

    for room_name in "${room_names[@]}"; do
        if should_skip_room_name "${room_name}"; then
            continue
        fi

        if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
            continue
        fi

        (( eligible_room_count += 1 ))

        if is_room_nonempty "${OUTPUT_ROOT}/${scene_name}/${room_name}"; then
            (( processed_room_count += 1 ))
        fi
    done

    SCENE_COUNT_ELIGIBLE_RESULT="${eligible_room_count}"
    SCENE_COUNT_PROCESSED_RESULT="${processed_room_count}"
}

is_scene_complete() {
    local scene_name="$1"
    shift
    local room_names=("$@")

    scene_processed_counts "${scene_name}" "${room_names[@]}"

    if (( SCENE_COUNT_ELIGIBLE_RESULT == 0 )); then
        return 0
    fi

    (( SCENE_COUNT_PROCESSED_RESULT * 2 > SCENE_COUNT_ELIGIBLE_RESULT ))
}

echo "========================================================"
echo "  batch_counting_merge runner"
echo "  scenes_dir             : ${SCENES_DIR}"
echo "  room_objects           : ${ROOM_OBJECTS}"
echo "  keys_json              : ${KEYS_JSON}"
echo "  keys_clip_top3_json    : ${KEYS_CLIP_TOP3_JSON}"
echo "  max_per_case           : ${MAX_PER_CASE}"
echo "  parallel_scenes        : ${PARALLEL_SCENES}"
echo "  seed                   : ${SEED}"
echo "  output_root            : ${OUTPUT_ROOT}"
echo "  skip_render            : ${SKIP_RENDER}"
echo "  load_full_scene        : ${LOAD_FULL_SCENE}"
echo "  disable_runtime_physics: ${DISABLE_RUNTIME_PHYSICS}"
echo "  scene_filter           : '${SCENE_FILTER}'"
echo "  room_filter            : '${ROOM_FILTER}'"
echo "  omnigibson_appdata     : ${OMNIGIBSON_APPDATA_PATH}"
echo "  cache_cleanup_interval : ${CACHE_CLEANUP_INTERVAL}"
echo "  merge_script           : ${MERGE_SCRIPT##*/}"
echo "  task_types             :"
for task_type in "${TASK_TYPES_ARR[@]}"; do
    echo "    - ${task_type}"
done
echo "========================================================"

mkdir -p "${OMNIGIBSON_APPDATA_PATH}"

declare -a UNIQUE_SCENES=()
declare -A SCENE_SEEN=()
declare -A SCENE_PENDING_COUNTS=()
declare -A ROOMS_BY_SCENE=()
declare -A FLOOR_BY_SCENE_ROOM=()
declare -A SCENE_JOB_ROOMS=()
declare -A SCENE_JOB_FLOORS=()

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

for scene_dir in "${SCENES_DIR}"/*; do
    [[ -d "${scene_dir}" ]] || continue

    scene_name="${scene_dir##*/}"
    scene_has_pending_room=0
    scene_has_completed_room=0

    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    if [[ -z "${ROOMS_BY_SCENE["${scene_name}"]:-}" ]]; then
        rooms=()
    else
        mapfile -t rooms <<< "${ROOMS_BY_SCENE["${scene_name}"]}"
    fi

    if (( ${#rooms[@]} == 0 )); then
        echo "[SKIP SCENE] ${scene_name} - no rooms found in ROOM_OBJECTS"
        continue
    fi

    if is_scene_complete "${scene_name}" "${rooms[@]}"; then
        echo "[SKIP SCENE] ${scene_name} - non-empty room outputs reach half (${SCENE_COUNT_PROCESSED_RESULT}/${SCENE_COUNT_ELIGIBLE_RESULT})"
        continue
    fi

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
            scene_has_completed_room=1
            continue
        fi

        floor_name="${FLOOR_BY_SCENE_ROOM["${scene_name}"$'\t'"${room_name}"]:-}"

        if [[ -z "${floor_name}" ]]; then
            echo "[SKIP] ${scene_name} / ${room_name} - no floor found"
            continue
        fi

        if [[ -z "${SCENE_SEEN[$scene_name]:-}" ]]; then
            UNIQUE_SCENES+=("${scene_name}")
            SCENE_SEEN["${scene_name}"]=1
            SCENE_PENDING_COUNTS["${scene_name}"]=0
            SCENE_JOB_ROOMS["${scene_name}"]=""
            SCENE_JOB_FLOORS["${scene_name}"]=""
        fi
        SCENE_PENDING_COUNTS["${scene_name}"]=$(( ${SCENE_PENDING_COUNTS["${scene_name}"]} + 1 ))
        if [[ -n "${SCENE_JOB_ROOMS["${scene_name}"]}" ]]; then
            SCENE_JOB_ROOMS["${scene_name}"]+=$'\n'
            SCENE_JOB_FLOORS["${scene_name}"]+=$'\n'
        fi
        SCENE_JOB_ROOMS["${scene_name}"]+="${room_name}"
        SCENE_JOB_FLOORS["${scene_name}"]+="${floor_name}"
        scene_has_pending_room=1
    done

    if [[ "${scene_has_pending_room}" -eq 0 && "${scene_has_completed_room}" -eq 1 ]]; then
        scene_processed_counts "${scene_name}" "${rooms[@]}"
        echo "[SKIP SCENE] ${scene_name} - non-empty room outputs reach half (${SCENE_COUNT_PROCESSED_RESULT}/${SCENE_COUNT_ELIGIBLE_RESULT})"
    fi
done

if (( ${#UNIQUE_SCENES[@]} > 1 )); then
    reversed_scenes=()
    for (( idx=${#UNIQUE_SCENES[@]}-1; idx>=0; idx-- )); do
        reversed_scenes+=("${UNIQUE_SCENES[idx]}")
    done
    UNIQUE_SCENES=("${reversed_scenes[@]}")
fi

n_rooms=0
for scene_name in "${UNIQUE_SCENES[@]}"; do
    n_rooms=$(( n_rooms + ${SCENE_PENDING_COUNTS["${scene_name}"]:-0} ))
done
scene_jobs_total="${#UNIQUE_SCENES[@]}"
num_tasks="${#TASK_TYPES_ARR[@]}"
total_task_outputs=$(( n_rooms * num_tasks ))
echo "  rooms found            : ${n_rooms}"
echo "  scene jobs             : ${scene_jobs_total}"
echo "  task outputs           : ${total_task_outputs}"
echo "========================================================"

if [[ "${n_rooms}" -eq 0 ]]; then
    echo "[DONE] No pending rooms found. All matching scenes / rooms are already complete."
    exit 0
fi

total=0
success=0
fail=0
skip=0
script_runs_since_cleanup=0
scene_jobs_completed=0
last_progress_scene=""
TMP_STATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/batch_counting_merge.XXXXXX")"
trap 'rm -rf "${TMP_STATE_DIR}"' EXIT

declare -a ACTIVE_PIDS=()
declare -A PID_TO_SCENE=()
declare -A PID_TO_LOG=()
declare -A PID_TO_APPDATA=()

process_scene_job_result() {
    local pid="$1"
    local exit_code="$2"
    local scene_name="${PID_TO_SCENE[$pid]}"
    local log_file="${PID_TO_LOG[$pid]}"
    local scene_appdata_path="${PID_TO_APPDATA[$pid]}"
    local active_after_completion="${#ACTIVE_PIDS[@]}"
    local room_name
    local task_type
    local room_root
    local room_success=1
    local -a scene_rooms=()

    if (( active_after_completion > 0 )); then
        (( active_after_completion-- ))
    fi

    mapfile -t scene_rooms <<< "${SCENE_JOB_ROOMS["${scene_name}"]}"
    for room_name in "${scene_rooms[@]}"; do
        room_root="${OUTPUT_ROOT}/${scene_name}/${room_name}"
        room_success=1

        if is_room_attempted "${room_root}"; then
            echo "[SKIP ROOM] ${scene_name} / ${room_name} - room bbox area outside threshold or room previously marked attempted"
            skip=$(( skip + num_tasks ))
            continue
        fi

        for task_type in "${TASK_TYPES_ARR[@]}"; do
            if is_task_complete "${room_root}" "${task_type}"; then
                echo "[OK] ${scene_name} / ${room_name} / ${task_type}"
                (( success++ ))
            else
                if [[ "${exit_code}" -ne 0 ]]; then
                    echo "[ERROR] ${scene_name} / ${room_name} / ${task_type} - scene exit=${exit_code}"
                else
                    echo "[MISSING] ${scene_name} / ${room_name} / ${task_type} - no JSON generated"
                fi
                (( fail++ ))
                room_success=0
            fi
        done

        if [[ "${room_success}" -eq 1 ]]; then
            echo "[OK] room complete: ${scene_name} / ${room_name}"
        fi
    done

    (( script_runs_since_cleanup++ ))
    if (( CACHE_CLEANUP_INTERVAL > 0 && script_runs_since_cleanup >= CACHE_CLEANUP_INTERVAL )); then
        cleanup_omnigibson_cache "${scene_appdata_path}"
        script_runs_since_cleanup=0
    fi

    (( scene_jobs_completed++ ))
    last_progress_scene="${scene_name}"
    render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" "${active_after_completion}" "completed" "${scene_name}"

    unset PID_TO_SCENE["$pid"] PID_TO_LOG["$pid"] PID_TO_APPDATA["$pid"]
}

reap_room_jobs() {
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

            process_scene_job_result "${pid}" "${exit_code}"
            completed_any=1
        done

        ACTIVE_PIDS=("${remaining_pids[@]}")

        if (( completed_any == 1 || wait_for_one == 0 )); then
            if (( wait_for_one == 1 || ${#ACTIVE_PIDS[@]} == 0 )); then
                break
            fi
        fi

        sleep 1
    done
}

render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" 0 "queued" ""

for scene_name in "${UNIQUE_SCENES[@]}"; do
    py_output_root="${OUTPUT_ROOT}"
    mapfile -t scene_rooms <<< "${SCENE_JOB_ROOMS["${scene_name}"]}"
    mapfile -t scene_floors <<< "${SCENE_JOB_FLOORS["${scene_name}"]}"

    if (( ${#scene_rooms[@]} == 0 )); then
        continue
    fi

    clear_progress_line
    echo ""
    echo "----------------------------------------------------"
    echo "  scene=${scene_name}"
    echo "  rooms=${#scene_rooms[@]}"
    echo "  seed =${SEED}"
    echo "  root =${py_output_root}/${scene_name}"
    echo "----------------------------------------------------"

    total=$(( total + ${#scene_rooms[@]} * num_tasks ))

    log_dir="${OUTPUT_ROOT}/logs/${scene_name}"
    mkdir -p "${log_dir}"
    log_file="${log_dir}/${scene_name}_counting_merge.log"

    echo "  script=${MERGE_SCRIPT##*/}"
    echo "  tasks =${TASK_TYPES_ARR[*]}"
    echo "  rooms =${scene_rooms[*]}"
    echo "  log   =${log_file}"

    worker_appdata_path="${TMP_STATE_DIR}/appdata_$(sanitize_path_component "${scene_name}")"
    mkdir -p "${worker_appdata_path}"

    cmd=(
        python3 "${MERGE_SCRIPT}"
        --scene "${scene_name}"
        --seed "${SEED}"
        --keys_json "${KEYS_JSON}"
        --keys_clip_top3_json "${KEYS_CLIP_TOP3_JSON}"
        --robot "${ROBOT}"
        --max_per_case "${MAX_PER_CASE}"
        --output_root "${py_output_root}"
        --rooms "${scene_rooms[@]}"
    )
    if (( ${#scene_floors[@]} > 0 )); then
        cmd+=(--floors "${scene_floors[@]}")
    fi

    if [[ "${SKIP_RENDER}" == "1" ]]; then
        cmd+=(--skip_render)
    fi
    if [[ "${LOAD_FULL_SCENE}" == "1" ]]; then
        cmd+=(--load_full_scene)
    fi
    if [[ "${DISABLE_RUNTIME_PHYSICS}" == "1" ]]; then
        cmd+=(--disable_runtime_physics)
    fi
    if (( ${#TASK_TYPES_ARR[@]} > 0 )); then
        cmd+=(--task_types "${TASK_TYPES_ARR[@]}")
    fi

    OMNIGIBSON_APPDATA_PATH="${worker_appdata_path}" "${cmd[@]}" > "${log_file}" 2>&1 &
    pid=$!
    ACTIVE_PIDS+=("${pid}")
    PID_TO_SCENE["${pid}"]="${scene_name}"
    PID_TO_LOG["${pid}"]="${log_file}"
    PID_TO_APPDATA["${pid}"]="${worker_appdata_path}"
    render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" "${#ACTIVE_PIDS[@]}" "running" "${scene_name}"

    while (( ${#ACTIVE_PIDS[@]} >= PARALLEL_SCENES )); do
        reap_room_jobs 1
    done
done

reap_room_jobs 0

echo ""
echo "========================================================"
echo "  Done."
echo "  total=${total}  ok=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"
render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" 0 "done" "${last_progress_scene}"
