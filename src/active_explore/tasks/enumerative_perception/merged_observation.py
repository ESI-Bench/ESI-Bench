from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "merged_observation"
FULL_SCENE = True
TASK_LABEL = "Merged Observation"
TASK_CASE = "observation_merged"
TASK_FOCUS = "Merge partially overlapping views into a single object set while avoiding duplicate target instances."
TASK_RULES = [
    "This case merges evidence across views.",
    "Union the visible instances across viewpoints without double-counting the same object.",
]


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(
        payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES
    )


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)
