---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/questions.jsonl
---

# BEHAVIOR ESI-Bench

BEHAVIOR ESI-Bench is a question dataset for evaluating embodied spatial intelligence across indoor scenes, object arrangements, physical reasoning, temporal understanding, and active exploration tasks.

Each row is one valid question instance. The table is intentionally flattened for Hugging Face Dataset Viewer and Croissant compatibility.

## Schema

```text
id
big_task
small_task
runner_task
scene
room
question
answer
answer_type
options_json
image_paths_json
metadata_json
```

`options_json`, `image_paths_json`, and `metadata_json` are JSON-encoded strings. `metadata_json` stores the task payload needed by the original runner, with duplicated top-level row fields and empty values removed.

The mirrored local files under `dataset/json_clean` use this same top-level schema. The older `dataset/json` tree is the raw source used to regenerate this table.

## Task Taxonomy

The dataset follows the ESI-Bench table hierarchy:

```text
Action Sequencing
Cognitive Mapping
Enumerative Perception
Metric Comparison
Perceptual Grounding
Physical Dynamics
Physical Structure
Spatial Relations
Specular Reflection
Temporal Understanding
```

The `small_task` column stores the corresponding subtask, and `runner_task` stores the internal task module name used by the original BEHAVIOR active-exploration code.

## Croissant

Hugging Face automatically generates Croissant metadata from the Dataset Viewer once this dataset is processed. The Croissant JSON-LD endpoint is:

```text
https://huggingface.co/api/datasets/ESI-Bench/esi-bench/croissant
```
