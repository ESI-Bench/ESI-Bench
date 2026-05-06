#!/usr/bin/env bash
#
# batch_cognitivemap_merge.sh
#
# Iterates over every scene found in SCENES_DIR, calling one merged
# batch_cognitivemap_merge.py script once per (scene, run_idx).
# The merged script generates connect / region / plan outputs in a single
# OmniGibson session so each scene only needs to be loaded once per run.
#
# Usage:
#   chmod +x batch_cognitivemap_merge.sh
#   ./batch_cognitivemap_merge.sh
#
# Optional env var overrides:
#   SCENES_DIR=datasets/behavior-1k-assets/scenes
#   OUTPUT_ROOT=renders_cognitivemap_batch
#   RUNS_PER_ROOM=1
#   PARALLEL_SCENES=2
#   SCENE_FILTER=Merom
#   TRAV_MAP_BASENAME=floor_trav_no_door
#   POINT_CANDIDATES=7
#   MAX_PAIR_CASES=8
#   MAX_TRIPLE_CASES=8
#   MAX_DIAGNOSTIC_PAIRS=64
#   CONNECT_REGION_LIMIT=8
#   SHORTEST_PATH_TIMEOUT_S=60
#   MAX_BELONG_CASES=16
#   MAX_SAME_REGION_CASES=16
#   MAX_CLOSER_REGION_CASES=16
#   MAX_PLAN_CASES=6
#   PLAN_REGION_LIMIT=8
#   REGION_EXPANSION_RATIO=0.35
#   REGION_EXPANSION_MIN=0.75
#   CLOSER_MARGIN=0.75
OMNIGIBSON_APPDATA_PATH=/data/$USER/omnigibson_appdata
CACHE_CLEANUP_INTERVAL=10

SCENES_DIR="${SCENES_DIR:-/home/jliu/Desktop/project/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-renders_cognitivemap_batch}"
RUNS_PER_ROOM="${RUNS_PER_ROOM:-1}"
PARALLEL_SCENES="${PARALLEL_SCENES:-2}"
SCENE_FILTER="${SCENE_FILTER:-}"
TRAV_MAP_BASENAME="${TRAV_MAP_BASENAME:-floor_trav_no_door}"
POINT_CANDIDATES="${POINT_CANDIDATES:-7}"
MAX_PAIR_CASES="${MAX_PAIR_CASES:-8}"
MAX_TRIPLE_CASES="${MAX_TRIPLE_CASES:-8}"
MAX_DIAGNOSTIC_PAIRS="${MAX_DIAGNOSTIC_PAIRS:-64}"
CONNECT_REGION_LIMIT="${CONNECT_REGION_LIMIT:-8}"
SHORTEST_PATH_TIMEOUT_S="${SHORTEST_PATH_TIMEOUT_S:-60}"
MAX_BELONG_CASES="${MAX_BELONG_CASES:-16}"
MAX_SAME_REGION_CASES="${MAX_SAME_REGION_CASES:-16}"
MAX_CLOSER_REGION_CASES="${MAX_CLOSER_REGION_CASES:-16}"
MAX_PLAN_CASES="${MAX_PLAN_CASES:-6}"
PLAN_REGION_LIMIT="${PLAN_REGION_LIMIT:-8}"
REGION_EXPANSION_RATIO="${REGION_EXPANSION_RATIO:-0.35}"
REGION_EXPANSION_MIN="${REGION_EXPANSION_MIN:-0.75}"
CLOSER_MARGIN="${CLOSER_MARGIN:-0.75}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OG_APPDATA_PATH="${SCRIPT_DIR}/OmniGibson/appdata"
if [[ -z "${OMNIGIBSON_APPDATA_PATH:-}" && -d /data ]]; then
    OMNIGIBSON_APPDATA_PATH="/data/${USER}/omnigibson_appdata"
fi
OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-${DEFAULT_OG_APPDATA_PATH}}"
CACHE_CLEANUP_INTERVAL="${CACHE_CLEANUP_INTERVAL:-0}"

MERGE_SCRIPT="${SCRIPT_DIR}/batch_cognitivemap_merge.py"
TASK_TYPES=(
    "pair_connectivity"
    "shortest_path_via_region"
    "object_in_region"
    "objects_same_region"
    "object_closer_region"
    "navigation_actions"
    "navigation_regions"
)
METADATA_FILES=(
    "cognitivemap_connect_candidates.json"
    "cognitivemap_region_candidates.json"
    "cognitivemap_plan_candidates.json"
)

run_output_root() {
    printf "%s" "${OUTPUT_ROOT}"
}

scene_run_dir() {
    local scene_name="$1"
    local run_idx="$2"
    printf "%s/%s/full_scene" "$(run_output_root "${run_idx}")" "${scene_name}"
}

cleanup_omnigibson_cache() {
    local appdata_root="${1:-${OMNIGIBSON_APPDATA_PATH}}"
    local cache_dir="${appdata_root}/global/cache"

    if [[ ! -d "${cache_dir}" ]]; then
        return 0
    fi

    echo "[CACHE] clearing ${cache_dir}"
    find "${cache_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
}

sanitize_path_component() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    value="${value//[^[:alnum:]_.-]/_}"
    printf '%s\n' "${value}"
}

is_task_complete() {
    local scene_run_dir="$1"
    local scene_name="$2"
    local task_type="$3"
    local question_root="${scene_run_dir}/cognitivemap_question_jsons"

    [[ -d "${question_root}" ]] || return 1

    python3 -c "
import json, pathlib, sys

question_root = pathlib.Path(sys.argv[1])
target_task_type = sys.argv[2]

for json_path in question_root.rglob('*.json'):
    try:
        with json_path.open() as f:
            payload = json.load(f)
    except Exception:
        continue
    if payload.get('task_type') == target_task_type:
        sys.exit(0)

sys.exit(1)
" "${question_root}" "${task_type}" >/dev/null 2>&1
}

is_connect_task_skipped() {
    local scene_run_dir="$1"
    local scene_name="$2"
    local metadata_path="${scene_run_dir}/cognitivemap_connect_candidates.json"

    [[ -f "${metadata_path}" ]] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
sys.exit(0 if payload.get('skipped') else 1)
" "${metadata_path}" >/dev/null 2>&1
}

is_plan_task_skipped() {
    local scene_run_dir="$1"
    local scene_name="$2"
    local metadata_path="${scene_run_dir}/cognitivemap_plan_candidates.json"

    [[ -f "${metadata_path}" ]] || return 1
    python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
sys.exit(0 if payload.get('skipped') else 1)
" "${metadata_path}" >/dev/null 2>&1
}

is_metadata_complete() {
    local scene_run_dir="$1"
    local scene_name="$2"
    local file_name="$3"
    local metadata_path="${scene_run_dir}/${file_name}"

    [[ -f "${metadata_path}" ]]
}

is_scene_run_complete() {
    local scene_name="$1"
    local run_idx="$2"
    local scene_dir
    scene_dir="$(scene_run_dir "${scene_name}" "${run_idx}")"

    for task_type in "${TASK_TYPES[@]}"; do
        if [[ "${task_type}" == "pair_connectivity" || "${task_type}" == "shortest_path_via_region" ]]; then
            if is_connect_task_skipped "${scene_dir}" "${scene_name}"; then
                continue
            fi
        fi
        if [[ "${task_type}" == "navigation_actions" || "${task_type}" == "navigation_regions" ]]; then
            if is_plan_task_skipped "${scene_dir}" "${scene_name}"; then
                continue
            fi
        fi
        if ! is_task_complete "${scene_dir}" "${scene_name}" "${task_type}"; then
            return 1
        fi
    done

    for metadata_file in "${METADATA_FILES[@]}"; do
        if ! is_metadata_complete "${scene_dir}" "${scene_name}" "${metadata_file}"; then
            return 1
        fi
    done

    return 0
}

is_scene_complete() {
    local scene_name="$1"

    for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
        if ! is_scene_run_complete "${scene_name}" "${run_idx}"; then
            return 1
        fi
    done

    return 0
}

echo "========================================================"
echo "  batch_cognitivemap_merge runner"
echo "  scenes_dir              : ${SCENES_DIR}"
echo "  output_root             : ${OUTPUT_ROOT}"
echo "  runs_per_room           : ${RUNS_PER_ROOM}"
echo "  parallel_scenes         : ${PARALLEL_SCENES}"
echo "  scene_filter            : '${SCENE_FILTER}'"
echo "  trav_map_basename       : ${TRAV_MAP_BASENAME}"
echo "  point_candidates        : ${POINT_CANDIDATES}"
echo "  max_pair_cases          : ${MAX_PAIR_CASES}"
echo "  max_triple_cases        : ${MAX_TRIPLE_CASES}"
echo "  max_diagnostic_pairs    : ${MAX_DIAGNOSTIC_PAIRS}"
echo "  connect_region_limit    : ${CONNECT_REGION_LIMIT}"
echo "  shortest_path_timeout_s : ${SHORTEST_PATH_TIMEOUT_S}"
echo "  max_belong_cases        : ${MAX_BELONG_CASES}"
echo "  max_same_region_cases   : ${MAX_SAME_REGION_CASES}"
echo "  max_closer_region_cases : ${MAX_CLOSER_REGION_CASES}"
echo "  max_plan_cases          : ${MAX_PLAN_CASES}"
echo "  plan_region_limit       : ${PLAN_REGION_LIMIT}"
echo "  region_expansion_ratio  : ${REGION_EXPANSION_RATIO}"
echo "  region_expansion_min    : ${REGION_EXPANSION_MIN}"
echo "  closer_margin           : ${CLOSER_MARGIN}"
echo "  omnigibson_appdata      : ${OMNIGIBSON_APPDATA_PATH}"
echo "  cache_cleanup_interval  : ${CACHE_CLEANUP_INTERVAL}"
echo "  merge_script            : ${MERGE_SCRIPT##*/}"
echo "  task_types              :"
for task_type in "${TASK_TYPES[@]}"; do
    echo "    - ${task_type}"
done
echo "========================================================"

mkdir -p "${OMNIGIBSON_APPDATA_PATH}"

declare -a SCENE_NAMES=()

for scene_dir in "${SCENES_DIR}"/*; do
    [[ -d "${scene_dir}" ]] || continue

    scene_name="${scene_dir##*/}"

    if [[ -n "${SCENE_FILTER}" && "${scene_name}" != *"${SCENE_FILTER}"* ]]; then
        continue
    fi

    if is_scene_complete "${scene_name}"; then
        echo "[SKIP SCENE] ${scene_name} - all runs already complete"
        continue
    fi

    SCENE_NAMES+=("${scene_name}")
done

n_scenes="${#SCENE_NAMES[@]}"
num_tasks="${#TASK_TYPES[@]}"
num_metadata="${#METADATA_FILES[@]}"
total_scene_runs=$(( n_scenes * RUNS_PER_ROOM ))
total_task_outputs=$(( total_scene_runs * num_tasks ))
echo "  scenes found            : ${n_scenes}"
echo "  scene runs              : ${total_scene_runs}"
echo "  task outputs            : ${total_task_outputs}"
echo "========================================================"

if [[ "${n_scenes}" -eq 0 ]]; then
    echo "[DONE] No pending scenes found. All matching scenes are already complete."
    exit 0
fi

total=0
success=0
fail=0
skip=0
script_runs_since_cleanup=0
TMP_STATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/batch_cognitivemap_merge.XXXXXX")"
trap 'rm -rf "${TMP_STATE_DIR}"' EXIT

declare -a ACTIVE_PIDS=()
declare -A PID_TO_SCENE=()
declare -A PID_TO_RUN=()
declare -A PID_TO_LOG=()
declare -A PID_TO_APPDATA=()

process_scene_job_result() {
    local pid="$1"
    local exit_code="$2"
    local scene_name="${PID_TO_SCENE[$pid]}"
    local run_idx="${PID_TO_RUN[$pid]}"
    local actual_scene_run_dir
    local task_type
    local metadata_file
    local run_success=1
    local connect_skipped=0
    local plan_skipped=0

    actual_scene_run_dir="$(scene_run_dir "${scene_name}" "${run_idx}")"

    if [[ "${exit_code}" -eq 0 ]]; then
        if is_connect_task_skipped "${actual_scene_run_dir}" "${scene_name}"; then
            connect_skipped=1
            echo "[SKIP] ${scene_name} / full_scene / run=${run_idx} / connect task intentionally skipped"
        fi
        if is_plan_task_skipped "${actual_scene_run_dir}" "${scene_name}"; then
            plan_skipped=1
            echo "[SKIP] ${scene_name} / full_scene / run=${run_idx} / plan task intentionally skipped"
        fi
        for task_type in "${TASK_TYPES[@]}"; do
            if [[ "${connect_skipped}" -eq 1 && ( "${task_type}" == "pair_connectivity" || "${task_type}" == "shortest_path_via_region" ) ]]; then
                (( skip++ ))
                continue
            fi
            if [[ "${plan_skipped}" -eq 1 && ( "${task_type}" == "navigation_actions" || "${task_type}" == "navigation_regions" ) ]]; then
                (( skip++ ))
                continue
            fi
            if is_task_complete "${actual_scene_run_dir}" "${scene_name}" "${task_type}"; then
                echo "[OK] ${scene_name} / full_scene / run=${run_idx} / ${task_type}"
                (( success++ ))
            else
                echo "[MISSING] ${scene_name} / full_scene / run=${run_idx} / ${task_type} - no JSON generated"
                (( fail++ ))
                run_success=0
            fi
        done
        for metadata_file in "${METADATA_FILES[@]}"; do
            if ! is_metadata_complete "${actual_scene_run_dir}" "${scene_name}" "${metadata_file}"; then
                echo "[MISSING] ${scene_name} / full_scene / run=${run_idx} / ${metadata_file}"
                (( fail++ ))
                run_success=0
            fi
        done
        if [[ "${run_success}" -eq 1 ]]; then
            echo "[OK] merged run complete: ${scene_name} / full_scene / run=${run_idx}"
        fi
    else
        for task_type in "${TASK_TYPES[@]}"; do
            echo "[ERROR] ${scene_name} / full_scene / run=${run_idx} / ${task_type} - exit=${exit_code}"
            (( fail++ ))
        done
    fi

    (( script_runs_since_cleanup++ ))
    if (( CACHE_CLEANUP_INTERVAL > 0 && script_runs_since_cleanup >= CACHE_CLEANUP_INTERVAL )); then
        cleanup_omnigibson_cache "${PID_TO_APPDATA[$pid]}"
        script_runs_since_cleanup=0
    fi

    unset PID_TO_SCENE["$pid"] PID_TO_RUN["$pid"] PID_TO_LOG["$pid"] PID_TO_APPDATA["$pid"]
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

for (( run_idx=0; run_idx<RUNS_PER_ROOM; run_idx++ )); do
    echo ""
    echo "###################################################"
    echo "  run_idx=${run_idx}  (across ${n_scenes} scenes)"
    echo "###################################################"

    for (( i=0; i<n_scenes; i++ )); do
        scene_name="${SCENE_NAMES[$i]}"
        py_output_root="$(run_output_root "${run_idx}")"
        actual_scene_run_dir="$(scene_run_dir "${scene_name}" "${run_idx}")"
        seed=$(( run_idx + 1 ))

        if is_scene_run_complete "${scene_name}" "${run_idx}"; then
            echo ""
            echo "[SKIP RUN] ${scene_name} / full_scene / run=${run_idx} - all tasks already complete"
            skip=$(( skip + num_tasks ))
            continue
        fi

        echo ""
        echo "----------------------------------------------------"
        echo "  scene=${scene_name}"
        echo "  room =full_scene"
        echo "  run  =${run_idx}"
        echo "  seed =${seed}"
        echo "  root =${py_output_root}"
        echo "  dir  =${actual_scene_run_dir}"
        echo "----------------------------------------------------"

        (( total += num_tasks ))

        log_dir="${OUTPUT_ROOT}/logs/${scene_name}/full_scene"
        mkdir -p "${log_dir}"
        log_file="${log_dir}/run_$(printf '%04d' "${run_idx}")_cognitivemap_merge.log"

        echo "  script=${MERGE_SCRIPT##*/}"
        echo "  tasks =${TASK_TYPES[*]}"
        echo "  log   =${log_file}"

        worker_appdata_path="${TMP_STATE_DIR}/appdata_$(sanitize_path_component "${scene_name}")_run_$(printf '%04d' "${run_idx}")"
        mkdir -p "${worker_appdata_path}"

        cmd=(
            python3 "${MERGE_SCRIPT}"
            --scene "${scene_name}"
            --seed "${seed}"
            --point_candidates "${POINT_CANDIDATES}"
            --trav_map_basename "${TRAV_MAP_BASENAME}"
            --output_root "${py_output_root}"
            --max_pair_cases "${MAX_PAIR_CASES}"
            --max_triple_cases "${MAX_TRIPLE_CASES}"
            --max_diagnostic_pairs "${MAX_DIAGNOSTIC_PAIRS}"
            --connect_region_limit "${CONNECT_REGION_LIMIT}"
            --shortest_path_timeout_s "${SHORTEST_PATH_TIMEOUT_S}"
            --max_belong_cases "${MAX_BELONG_CASES}"
            --max_same_region_cases "${MAX_SAME_REGION_CASES}"
            --max_closer_region_cases "${MAX_CLOSER_REGION_CASES}"
            --max_plan_cases "${MAX_PLAN_CASES}"
            --plan_region_limit "${PLAN_REGION_LIMIT}"
            --region_expansion_ratio "${REGION_EXPANSION_RATIO}"
            --region_expansion_min "${REGION_EXPANSION_MIN}"
            --closer_margin "${CLOSER_MARGIN}"
        )

        OMNIGIBSON_APPDATA_PATH="${worker_appdata_path}" "${cmd[@]}" > "${log_file}" 2>&1 &
        pid=$!
        ACTIVE_PIDS+=("${pid}")
        PID_TO_SCENE["${pid}"]="${scene_name}"
        PID_TO_RUN["${pid}"]="${run_idx}"
        PID_TO_LOG["${pid}"]="${log_file}"
        PID_TO_APPDATA["${pid}"]="${worker_appdata_path}"

        while (( ${#ACTIVE_PIDS[@]} >= PARALLEL_SCENES )); do
            reap_scene_jobs 1
        done
    done
done

reap_scene_jobs 0

echo ""
echo "========================================================"
echo "  Done."
echo "  total=${total}  ok=${success}  fail=${fail}  skip=${skip}"
echo "========================================================"
