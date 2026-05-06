#!/usr/bin/env bash
#
# run_storage.sh
#
# Mirrors original batch executor bash structure, calling batch_storage.py.
#
# Usage:
#   chmod +x run_storage.sh && ./run_storage.sh
#
# Env var overrides:
#   TASK_FOLDER, OUTPUT_BASE, OBJECT_INVENTORY, ROOM_OBJECTS,
#   ROBOT_TYPE, LOG_DIR, JSON_PATTERN, SCENE_FILTER

TASK_FOLDER="${TASK_FOLDER:-generated_data_storage_revised2}"
OUTPUT_BASE="${OUTPUT_BASE:-renders_storage_final}"
PYTHON_SCRIPT="batch_storage.py"
OBJECT_INVENTORY="${OBJECT_INVENTORY:-bddl3/bddl/generated_data/object_inventory.json}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
ROBOT_TYPE="${ROBOT_TYPE:-R1}"
LOG_DIR="${LOG_DIR:-execution_logs_storage}"
JSON_PATTERN="${JSON_PATTERN:-*_revised.json}"
SCENE_FILTER="${SCENE_FILTER:-}"

mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "================================================"
echo "BEHAVIOR Storage Batch Executor"
echo "================================================"
echo "Task folder  : $TASK_FOLDER"
echo "Output       : $OUTPUT_BASE"
echo "Pattern      : $JSON_PATTERN"
echo "Scene filter : '${SCENE_FILTER}'"
echo "================================================"
echo ""

if [ ! -d "$TASK_FOLDER" ]; then
    echo -e "${RED}ERROR: Task folder '$TASK_FOLDER' not found!${NC}"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}ERROR: '$PYTHON_SCRIPT' not found!${NC}"
    exit 1
fi

TOTAL_TASKS=0
SUCCESSFUL_TASKS=0
FAILED_TASKS=0
SKIPPED_TASKS=0

for task_json in "$TASK_FOLDER"/$JSON_PATTERN; do

    if [ ! -f "$task_json" ]; then
        echo -e "${YELLOW}No files matching '$JSON_PATTERN' in $TASK_FOLDER${NC}"
        break
    fi

    task_name=$(basename "$task_json" .json)

    if [[ -n "$SCENE_FILTER" && "$task_name" != *"$SCENE_FILTER"* ]]; then
        continue
    fi

    TOTAL_TASKS=$((TOTAL_TASKS + 1))
    task_output_dir="$OUTPUT_BASE/$task_name"

    # Skip if output dir already exists and is non-empty
    if [ -d "$task_output_dir" ] && [ -n "$(ls -A "$task_output_dir" 2>/dev/null)" ]; then
        echo "================================================"
        echo -e "${CYAN}⊘ Skipping [$TOTAL_TASKS]: $task_name${NC}"
        echo -e "${CYAN}  Output already exists: $task_output_dir${NC}"
        echo "================================================"
        echo ""
        SKIPPED_TASKS=$((SKIPPED_TASKS + 1))
        continue
    fi

    log_file="$LOG_DIR/${task_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "================================================"
    echo -e "${YELLOW}Processing [$TOTAL_TASKS]: $task_name${NC}"
    echo "Input  : $task_json"
    echo "Output : $task_output_dir"
    echo "Log    : $log_file"
    echo "================================================"

    python "$PYTHON_SCRIPT" \
        --task_file        "$task_json" \
        --object-inventory "$OBJECT_INVENTORY" \
        --room-objects     "$ROOM_OBJECTS" \
        --robot            "$ROBOT_TYPE" \
        --output_root      "$OUTPUT_BASE" \
        2>&1 | tee "$log_file"

    exit_status=${PIPESTATUS[0]}

    if [ "$exit_status" -eq 0 ]; then
        echo -e "${GREEN}✓ $task_name — success${NC}"
        SUCCESSFUL_TASKS=$((SUCCESSFUL_TASKS + 1))
    elif [ "$exit_status" -eq 1 ]; then
        echo -e "${YELLOW}~ $task_name — partial (no successful placement / not visible)${NC}"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo -e "${YELLOW}Log: $log_file${NC}"
    else
        # Segfault (139) or other error — keep whatever output was written,
        # do NOT delete the output directory.
        echo -e "${RED}✗ $task_name — error (exit $exit_status), output preserved${NC}"
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo -e "${RED}Log: $log_file${NC}"
    fi

    echo ""
done

echo "================================================"
echo "EXECUTION SUMMARY"
echo "================================================"
echo "Total tasks      : $TOTAL_TASKS"
echo -e "${CYAN}Skipped          : $SKIPPED_TASKS${NC}"
echo -e "${GREEN}Successful       : $SUCCESSFUL_TASKS${NC}"
echo -e "${RED}Failed / Partial : $FAILED_TASKS${NC}"
echo ""
echo "Results : $OUTPUT_BASE"
echo "Logs    : $LOG_DIR"
echo "================================================"

[ $FAILED_TASKS -gt 0 ] && exit 1 || exit 0