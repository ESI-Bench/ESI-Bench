from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "structural_enclosure"
TASK_LABEL = "Structural Enclosure"
TASK_CASE = "hidden_in_box"
TASK_FOCUS = "Inspect boxes, microwaves, and other enclosures that can contain hidden target objects."
TASK_RULES = [
    "This case places targets inside enclosures or containers.",
    "Inspect container openings, doors, and hidden compartments carefully.",
]


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(
        payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES
    )


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)
