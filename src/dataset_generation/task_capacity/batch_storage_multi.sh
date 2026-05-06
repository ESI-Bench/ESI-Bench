#!/usr/bin/env bash
#
# run_storage_multi.sh
#
# Iterates over all task JSON files in TASK_FOLDER, calling
# batch_storage_multi.py once per file in a fresh Python process.
#
# Usage:
#   chmod +x run_storage_multi.sh && ./run_storage_multi.sh
#
# Optional env var overrides:
#   TASK_FOLDER, OUTPUT_BASE, KEYS_JSON, OBJECT_INVENTORY, ROOM_OBJECTS,
#   ROBOT_TYPE, LOG_DIR, JSON_PATTERN, SCENE_FILTER, NUM_OBJECTS

TASK_FOLDER="${TASK_FOLDER:-generated_data_storage_revised2}"
OUTPUT_BASE="${OUTPUT_BASE:-renders_storage_multi_final}"
PYTHON_SCRIPT="batch_storage_multi.py"
KEYS_JSON="${KEYS_JSON:-keys_fillable2.json}"
OBJECT_INVENTORY="${OBJECT_INVENTORY:-bddl3/bddl/generated_data/object_inventory.json}"
ROOM_OBJECTS="${ROOM_OBJECTS:-bddl3/bddl/generated_data/combined_room_object_list_future.json}"
ROBOT_TYPE="${ROBOT_TYPE:-R1}"
LOG_DIR="${LOG_DIR:-execution_logs_storage_multi}"
JSON_PATTERN="${JSON_PATTERN:-*_revised.json}"
SCENE_FILTER="${SCENE_FILTER:-}"
NUM_OBJECTS="${NUM_OBJECTS:-3}"   # 3 or 4

mkdir -p "$OUTPUT_BASE"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "========================================================"
echo "  Storage Multi Batch Runner"
echo "  task_folder  : ${TASK_FOLDER}"
echo "  output_root  : ${OUTPUT_BASE}"
echo "  keys_json    : ${KEYS_JSON}"
echo "  num_objects  : ${NUM_OBJECTS}"
echo "  pattern      : ${JSON_PATTERN}"
echo "  scene_filter : '${SCENE_FILTER}'"
echo "========================================================"
echo ""

if [ ! -d "$TASK_FOLDER" ]; then
    echo -e "${RED}ERROR: Task folder '$TASK_FOLDER' not found!${NC}"
    exit 1
fi
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}ERROR: '$PYTHON_SCRIPT' not found!${NC}"
    exit 1
fi
if [ ! -f "$KEYS_JSON" ]; then
    echo -e "${RED}ERROR: keys json '$KEYS_JSON' not found!${NC}"
    exit 1
fi

TOTAL=0
SUCCESS=0
PARTIAL=0
FAIL=0
SKIP=0

for task_json in "$TASK_FOLDER"/$JSON_PATTERN; do

    if [ ! -f "$task_json" ]; then
        echo -e "${YELLOW}No files matching '$JSON_PATTERN' in $TASK_FOLDER${NC}"
        break
    fi

    task_name=$(basename "$task_json" .json | sed 's/\.json//g' | sed 's/^_//;s/_$//')

    if [[ -n "$SCENE_FILTER" && "$task_name" != *"$SCENE_FILTER"* ]]; then
        continue
    fi

    (( TOTAL++ ))
    out_dir="${OUTPUT_BASE}/${task_name}"

    # Skip if output dir already exists and is non-empty
    if [ -d "$out_dir" ] && [ -n "$(ls -A "$out_dir" 2>/dev/null)" ]; then
        echo "----------------------------------------"
        echo -e "${CYAN}[SKIP] ${task_name}${NC}"
        (( SKIP++ ))
        continue
    fi

    log_file="${LOG_DIR}/${task_name}_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "========================================================"
    echo -e "${YELLOW}[${TOTAL}] ${task_name}${NC}"
    echo "  input  : ${task_json}"
    echo "  output : ${out_dir}"
    echo "  log    : ${log_file}"
    echo "========================================================"

    python "$PYTHON_SCRIPT" \
        --task_file        "$task_json" \
        --keys_json        "$KEYS_JSON" \
        --object-inventory "$OBJECT_INVENTORY" \
        --room-objects     "$ROOM_OBJECTS" \
        --robot            "$ROBOT_TYPE" \
        --output_root      "$OUTPUT_BASE" \
        2>&1 | tee "$log_file"

    EXIT_CODE=${PIPESTATUS[0]}

    case "$EXIT_CODE" in
        0)
            echo -e "${GREEN}[OK]      ${task_name}${NC}"
            (( SUCCESS++ ))
            ;;
        1)
            echo -e "${YELLOW}[PARTIAL] ${task_name} — no successful placement${NC}"
            (( PARTIAL++ ))
            ;;
        *)
            echo -e "${RED}[ERROR]   ${task_name} — exit ${EXIT_CODE}, output preserved${NC}"
            (( FAIL++ ))
            ;;
    esac

done

echo ""
echo "========================================================"
echo "  DONE"
echo "  total   : ${TOTAL}"
echo -e "  ${CYAN}skipped : ${SKIP}${NC}"
echo -e "  ${GREEN}success : ${SUCCESS}${NC}"
echo -e "  ${YELLOW}partial : ${PARTIAL}${NC}"
echo -e "  ${RED}error   : ${FAIL}${NC}"
echo "========================================================"

[ $FAIL -gt 0 ] && exit 1 || exit 0