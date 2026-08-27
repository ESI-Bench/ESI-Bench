from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "illumination_variability"
FULL_SCENE = True
TASK_LABEL = "Illumination Variability"
TASK_CASE = "light_change"
TASK_FOCUS = "Track the same target instances across normal and changed lighting without creating duplicate counts."
TASK_RULES = [
    "This case changes lighting across views.",
    "Do not treat lighting shifts as new objects; keep identity stable across illumination changes.",
]


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(
        payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES
    )


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)
