from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "counting_w_occlusion"
TASK_LABEL = "Counting w Occlusion"
TASK_CASE = "hidden_by_others"
TASK_FOCUS = "Recover targets hidden behind other objects by checking alternative viewpoints and occlusion boundaries."
TASK_RULES = [
    "This case hides targets behind other objects.",
    "Inspect occluded regions from fresh angles before finalizing the count.",
]


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(
        payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES
    )


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)
