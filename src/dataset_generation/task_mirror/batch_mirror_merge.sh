#!/usr/bin/env bash
#
# batch_mirror_merge.sh
#
# Iterates over every (scene, room) pair found in SCENES_DIR, calling
# one merged batch_mirror_merge.py script once per (room, run_idx).
# The merged script generates all mirror question types in a single
# OmniGibson session so each room only needs to be rendered once per run.
#
# Loop order: run_idx is the OUTER loop; scenes are scheduled within each
# run_idx and can execute in parallel. This means run 0 completes across
# all rooms before run 1 begins, while multiple scenes from the same run
# may be processed at the same time.
#
# Usage:
#   chmod +x batch_mirror_merge.sh
#   ./batch_mirror_merge.sh
#
# Optional env var overrides:
#   SCENES_DIR=scenes5
#   ROOM_OBJECTS=bddl3/bddl/generated_data/combined_room_object_list_future.json
#   KEYS_JSON=keys.json
#   ROBOT=R1
#   OUTPUT_ROOT=renders_mirror_batch
#   RUNS_PER_ROOM=5
#   MAX_QUESTIONS_PER_TYPE=8
#   PARALLEL_SCENES=2
#   SCENE_FILTER=Merom
#   ROOM_FILTER=bedroom
#   SKIP_RENDER=1
OMNIGIBSON_APPDATA_PATH=/data/$USER/omnigibson_appdata
CACHE_CLEANUP_INTERVAL=1

SCENES_DIR="${SCENES_DIR:-/home/jliu/Desktop/project/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
KEYS_JSON="${KEYS_JSON:-keys.json}"
ROBOT="${ROBOT:-R1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_mirror_batch}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
MAX_QUESTIONS_PER_TYPE="${MAX_QUESTIONS_PER_TYPE:-8}"
PARALLEL_SCENES="${PARALLEL_SCENES:-1}"
SCENE_FILTER="${SCENE_FILTER:-}"
ROOM_FILTER="${ROOM_FILTER:-}"
SKIP_RENDER="${SKIP_RENDER:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OG_APPDATA_PATH="${SCRIPT_DIR}/OmniGibson/appdata"
if [[ -z "${OMNIGIBSON_APPDATA_PATH:-}" && -d /data ]]; then
    OMNIGIBSON_APPDATA_PATH="/data/${USER}/omnigibson_appdata"
fi
OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-${DEFAULT_OG_APPDATA_PATH}}"
CACHE_CLEANUP_INTERVAL="${CACHE_CLEANUP_INTERVAL:-0}"

MERGE_SCRIPT="${SCRIPT_DIR}/batch_mirror_merge.py"
SKIPPED_ROOM_MARKER="mirror_room_skipped.json"
ATTEMPTED_ROOM_MARKER="mirror_room_attempted.json"
TASK_TYPES=(
    "mirror_object_reality"
    "mirror_distance"
    "mirror_correspondence"
)

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

    [[ "${room_name}" == *garden* || "${room_name}" == *corridor* ]]
}

is_task_complete() {
    local run_root="$1"
    local scene_name="$2"
    local task_type="$3"
    local output_task_dir="${run_root}/${scene_name}/mirror_question_jsons/${task_type}"
    local skip_marker_path="${run_root}/${scene_name}/${SKIPPED_ROOM_MARKER}"

    if [[ -f "${skip_marker_path}" ]]; then
        return 0
    fi

    [[ -d "${output_task_dir}" ]] && compgen -G "${output_task_dir}/*.json" > /dev/null
}

is_room_run_skipped() {
    local run_root="$1"
    local scene_name="$2"
    local skip_marker_path="${run_root}/${scene_name}/${SKIPPED_ROOM_MARKER}"

    [[ -f "${skip_marker_path}" ]]
}

is_room_run_attempted() {
    local run_root="$1"
    local scene_name="$2"
    local attempted_marker_path="${run_root}/${scene_name}/${ATTEMPTED_ROOM_MARKER}"

    [[ -f "${attempted_marker_path}" ]]
}

is_room_run_complete() {
    local scene_name="$1"
    local room_name="$2"
    local run_idx="$3"
    local run_root="${OUTPUT_ROOT}/${scene_name}/${room_name}/run_$(printf '%04d' "${run_idx}")"
    local task_type

    if is_room_run_skipped "${run_root}" "${scene_name}"; then
        return 0
    fi

    if is_room_run_attempted "${run_root}" "${scene_name}"; then
        return 0
    fi

    # Treat any existing task output as evidence that this room/run has
    # already been attempted, so we do not restart partial generations.
    for task_type in "${TASK_TYPES[@]}"; do
        if [[ -d "${run_root}/${scene_name}/mirror_question_jsons/${task_type}" ]] && \
           compgen -G "${run_root}/${scene_name}/mirror_question_jsons/${task_type}/*.json" > /dev/null; then
            return 0
        fi
    done

    return 1
}

is_room_complete() {
    local scene_name="$1"
    local room_name="$2"

    for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
        if ! is_room_run_complete "${scene_name}" "${room_name}" "${run_idx}"; then
            return 1
        fi
    done

    return 0
}

is_scene_complete() {
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

        if is_room_complete "${scene_name}" "${room_name}"; then
            (( processed_room_count += 1 ))
        fi
    done

    if (( eligible_room_count == 0 )); then
        return 0
    fi

    (( processed_room_count * 2 > eligible_room_count ))
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
    local current_run="${5:--}"
    local last_scene="${6:-}"

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
    local message="[SCENE PROGRESS] [${bar}${pad}] ${completed}/${total_jobs} (${percent}%%) active=${active_jobs} phase=${phase} run=${current_run}"
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

echo "========================================================"
echo "  batch_mirror_merge runner"
echo "  scenes_dir             : ${SCENES_DIR}"
echo "  room_objects           : ${ROOM_OBJECTS}"
echo "  keys_json              : ${KEYS_JSON}"
echo "  runs_per_room          : ${RUNS_PER_ROOM}"
echo "  max_questions_per_type : ${MAX_QUESTIONS_PER_TYPE}"
echo "  parallel_scenes        : ${PARALLEL_SCENES}"
echo "  output_root            : ${OUTPUT_ROOT}"
echo "  skip_render            : ${SKIP_RENDER}"
echo "  scene_filter           : '${SCENE_FILTER}'"
echo "  room_filter            : '${ROOM_FILTER}'"
echo "  omnigibson_appdata     : ${OMNIGIBSON_APPDATA_PATH}"
echo "  cache_cleanup_interval : ${CACHE_CLEANUP_INTERVAL}"
echo "  merge_script           : ${MERGE_SCRIPT##*/}"
echo "  task_types             :"
for task_type in "${TASK_TYPES[@]}"; do
    echo "    - ${task_type}"
done
echo "========================================================"

mkdir -p "${OMNIGIBSON_APPDATA_PATH}"

declare -a SCENE_NAMES=()
declare -a ROOM_NAMES=()
declare -a FLOOR_NAMES=()
declare -a UNIQUE_SCENES=()
declare -A SCENE_SEEN=()

for scene_dir in "${SCENES_DIR}"/*; do
    [[ -d "${scene_dir}" ]] || continue

    scene_name="${scene_dir##*/}"
    scene_has_pending_room=0
    scene_has_completed_room=0

    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    mapfile -t rooms < <(python3 -c "
import json
with open('${ROOM_OBJECTS}') as f:
    data = json.load(f)
scenes = data.get('scenes', data)
for r in scenes.get('${scene_name}', {}).keys():
    print(r)
" 2>/dev/null)

    if (( ${#rooms[@]} == 0 )); then
        echo "[SKIP SCENE] ${scene_name} - no rooms found in ROOM_OBJECTS"
        continue
    fi

    if is_scene_complete "${scene_name}" "${rooms[@]}"; then
        eligible_room_count=0
        processed_room_count=0
        for room_name in "${rooms[@]}"; do
            if should_skip_room_name "${room_name}"; then
                continue
            fi
            if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
                continue
            fi
            (( eligible_room_count += 1 ))
            if is_room_complete "${scene_name}" "${room_name}"; then
                (( processed_room_count += 1 ))
            fi
        done
        echo "[SKIP SCENE] ${scene_name} - processed rooms exceed half (${processed_room_count}/${eligible_room_count})"
        continue
    fi

    for room_name in "${rooms[@]}"; do
        if should_skip_room_name "${room_name}"; then
            echo "[SKIP ROOM] ${scene_name} / ${room_name} - room name contains garden or corridor"
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
            echo "[SKIP] ${scene_name} / ${room_name} - no floor found"
            continue
        fi

        SCENE_NAMES+=("${scene_name}")
        ROOM_NAMES+=("${room_name}")
        FLOOR_NAMES+=("${floor_name}")
        if [[ -z "${SCENE_SEEN[$scene_name]:-}" ]]; then
            UNIQUE_SCENES+=("${scene_name}")
            SCENE_SEEN["${scene_name}"]=1
        fi
        scene_has_pending_room=1
    done

    if [[ "${scene_has_pending_room}" -eq 0 && "${scene_has_completed_room}" -eq 1 ]]; then
        eligible_room_count=0
        processed_room_count=0
        for room_name in "${rooms[@]}"; do
            if should_skip_room_name "${room_name}"; then
                continue
            fi
            if [[ -n "${ROOM_FILTER}" && "${room_name}" != *"${ROOM_FILTER}"* ]]; then
                continue
            fi
            (( eligible_room_count += 1 ))
            if is_room_complete "${scene_name}" "${room_name}"; then
                (( processed_room_count += 1 ))
            fi
        done
        echo "[SKIP SCENE] ${scene_name} - processed rooms exceed half (${processed_room_count}/${eligible_room_count})"
    fi
done

n_rooms="${#SCENE_NAMES[@]}"
num_tasks="${#TASK_TYPES[@]}"
total_room_runs=$(( n_rooms * RUNS_PER_ROOM ))
total_task_outputs=$(( total_room_runs * num_tasks ))
echo "  rooms found            : ${n_rooms}"
echo "  room runs              : ${total_room_runs}"
echo "  task outputs           : ${total_task_outputs}"
echo "========================================================"

if [[ "${n_rooms}" -eq 0 ]]; then
    echo "[DONE] No pending rooms found. All matching scenes / rooms are already complete."
    exit 0
fi

TMP_STATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/batch_mirror_merge.XXXXXX")"
trap 'rm -rf "${TMP_STATE_DIR}"' EXIT

total=0
success=0
fail=0
skip=0
script_runs_since_cleanup=0
scene_jobs_total=0
scene_jobs_completed=0

declare -a ACTIVE_PIDS=()
declare -A PID_TO_SCENE=()
declare -A PID_TO_RUN=()
declare -A PID_TO_ROOMS_FILE=()
declare -A PID_TO_APPDATA=()

process_scene_job_result() {
    local pid="$1"
    local exit_code="$2"
    local scene_name="${PID_TO_SCENE[$pid]}"
    local run_idx="${PID_TO_RUN[$pid]}"
    local rooms_file="${PID_TO_ROOMS_FILE[$pid]}"
    local scene_appdata_path="${PID_TO_APPDATA[$pid]}"
    local -a pending_room_names=()
    local active_after_completion="${#ACTIVE_PIDS[@]}"

    if (( active_after_completion > 0 )); then
        (( active_after_completion-- ))
    fi

    if [[ -f "${rooms_file}" ]]; then
        mapfile -t pending_room_names < "${rooms_file}"
    fi

    local room_name
    local py_output_root
    local run_success
    local task_type
    for room_name in "${pending_room_names[@]}"; do
        py_output_root="${OUTPUT_ROOT}/${scene_name}/${room_name}/run_$(printf '%04d' "${run_idx}")"

        if [[ "${exit_code}" -eq 0 ]]; then
            if is_room_run_skipped "${py_output_root}" "${scene_name}"; then
                for task_type in "${TASK_TYPES[@]}"; do
                    echo "[SKIP] ${scene_name} / ${room_name} / run=${run_idx} / ${task_type} - room skipped by merge script"
                    (( skip++ ))
                done
                echo "[SKIP] merged run skipped: ${scene_name} / ${room_name} / run=${run_idx}"
                continue
            fi

            run_success=1
            for task_type in "${TASK_TYPES[@]}"; do
                if is_task_complete "${py_output_root}" "${scene_name}" "${task_type}"; then
                    echo "[OK] ${scene_name} / ${room_name} / run=${run_idx} / ${task_type}"
                    (( success++ ))
                else
                    echo "[MISSING] ${scene_name} / ${room_name} / run=${run_idx} / ${task_type} - no JSON generated"
                    (( fail++ ))
                    run_success=0
                fi
            done
            if [[ "${run_success}" -eq 1 ]]; then
                echo "[OK] merged run complete: ${scene_name} / ${room_name} / run=${run_idx}"
            fi
        else
            for task_type in "${TASK_TYPES[@]}"; do
                echo "[ERROR] ${scene_name} / ${room_name} / run=${run_idx} / ${task_type} - exit=${exit_code}"
                (( fail++ ))
            done
        fi
    done

    (( script_runs_since_cleanup++ ))
    if (( CACHE_CLEANUP_INTERVAL > 0 && script_runs_since_cleanup >= CACHE_CLEANUP_INTERVAL )); then
        cleanup_omnigibson_cache "${scene_appdata_path}"
        script_runs_since_cleanup=0
    fi

    rm -f "${rooms_file}"
    unset PID_TO_SCENE["$pid"] PID_TO_RUN["$pid"] PID_TO_ROOMS_FILE["$pid"] PID_TO_APPDATA["$pid"]
    (( scene_jobs_completed++ ))
    render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" "${active_after_completion}" "completed" "${run_idx}" "${scene_name}"
}

reap_scene_jobs() {
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

for (( preview_run_idx=0; preview_run_idx<RUNS_PER_ROOM; preview_run_idx++ )); do
    for scene_name in "${UNIQUE_SCENES[@]}"; do
        scene_room_count=0
        for (( i=0; i<n_rooms; i++ )); do
            if [[ "${SCENE_NAMES[$i]}" != "${scene_name}" ]]; then
                continue
            fi
            room_name="${ROOM_NAMES[$i]}"
            if is_room_run_complete "${scene_name}" "${room_name}" "${preview_run_idx}"; then
                continue
            fi
            (( scene_room_count += 1 ))
        done
        if (( scene_room_count > 0 )); then
            (( scene_jobs_total += 1 ))
        fi
    done
done

echo "  scene jobs             : ${scene_jobs_total}"
render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" 0 "queued" "-" ""

for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
    clear_progress_line
    echo ""
    echo "###################################################"
    echo "  run_idx=${run_idx}  (across ${n_rooms} rooms)"
    echo "###################################################"

    for scene_name in "${UNIQUE_SCENES[@]}"; do
        seed=$(( run_idx + 1 ))
        scene_room_count=0
        declare -a PENDING_ROOM_NAMES=()
        cmd=(
            python3 "${MERGE_SCRIPT}"
            --scene "${scene_name}"
            --seed "${seed}"
            --run_idx "${run_idx}"
            --keys_json "${KEYS_JSON}"
            --robot "${ROBOT}"
            --max_questions_per_type "${MAX_QUESTIONS_PER_TYPE}"
            --output_root "${OUTPUT_ROOT}"
            --exit_on_finish
        )

        for (( i=0; i<n_rooms; i++ )); do
            if [[ "${SCENE_NAMES[$i]}" != "${scene_name}" ]]; then
                continue
            fi

            room_name="${ROOM_NAMES[$i]}"
            floor_name="${FLOOR_NAMES[$i]}"

            if is_room_run_complete "${scene_name}" "${room_name}" "${run_idx}"; then
                echo ""
                echo "[SKIP RUN] ${scene_name} / ${room_name} / run=${run_idx} - existing outputs detected"
                skip=$(( skip + num_tasks ))
                continue
            fi

            cmd+=(--room "${room_name}" --floor "${floor_name}")
            PENDING_ROOM_NAMES+=("${room_name}")
            (( total += num_tasks ))
            (( scene_room_count += 1 ))
        done

        if [[ "${scene_room_count}" -eq 0 ]]; then
            continue
        fi

        clear_progress_line
        echo ""
        echo "----------------------------------------------------"
        echo "  scene=${scene_name}"
        echo "  rooms=${scene_room_count}"
        echo "  run  =${run_idx}"
        echo "  seed =${seed}"
        echo "  root =${OUTPUT_ROOT}"
        echo "----------------------------------------------------"

        log_dir="${OUTPUT_ROOT}/logs/${scene_name}"
        mkdir -p "${log_dir}"
        log_file="${log_dir}/run_$(printf '%04d' "${run_idx}")_mirror_merge.log"
        rooms_file="${TMP_STATE_DIR}/$(sanitize_path_component "${scene_name}")_run_$(printf '%04d' "${run_idx}").rooms"
        printf '%s\n' "${PENDING_ROOM_NAMES[@]}" > "${rooms_file}"

        echo "  script=${MERGE_SCRIPT##*/}"
        echo "  tasks =${TASK_TYPES[*]}"
        echo "  log   =${log_file}"

        if [[ "${SKIP_RENDER}" == "1" ]]; then
            cmd+=(--skip_render)
        fi

        scene_appdata_path="${OMNIGIBSON_APPDATA_PATH}/scene_jobs/$(sanitize_path_component "${scene_name}")/run_$(printf '%04d' "${run_idx}")"
        mkdir -p "${scene_appdata_path}"
        echo "  appdata=${scene_appdata_path}"

        (
            OMNIGIBSON_APPDATA_PATH="${scene_appdata_path}" "${cmd[@]}" 2>&1 | tee "${log_file}"
            exit "${PIPESTATUS[0]}"
        ) &
        pid=$!

        ACTIVE_PIDS+=("${pid}")
        PID_TO_SCENE["${pid}"]="${scene_name}"
        PID_TO_RUN["${pid}"]="${run_idx}"
        PID_TO_ROOMS_FILE["${pid}"]="${rooms_file}"
        PID_TO_APPDATA["${pid}"]="${scene_appdata_path}"
        render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" "${#ACTIVE_PIDS[@]}" "started" "${run_idx}" "${scene_name}"

        if (( ${#ACTIVE_PIDS[@]} >= PARALLEL_SCENES )); then
            reap_scene_jobs 1
        fi
    done

    reap_scene_jobs 0
done

echo ""
echo "========================================================"
echo "  Done."
echo "  total=${total}  ok=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"
render_scene_progress "${scene_jobs_completed}" "${scene_jobs_total}" 0 "done" "-" ""
