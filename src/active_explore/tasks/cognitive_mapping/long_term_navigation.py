from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "long_term_navigation"
TASK_LABEL = "Long-Term Navigation"
TASK_FOCUS = "Plan or evaluate multi-step navigation across the full scene."


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS)


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS)
