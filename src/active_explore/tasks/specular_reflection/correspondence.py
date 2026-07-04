from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "correspondence"
TASK_LABEL = "Correspondence"
TASK_FOCUS = "Match real objects to their mirror reflections."


def build_system_prompt(payload, threshold, min_steps, camera_info=None, task_state=None):
    return common.build_system_prompt_for_task(payload, threshold, min_steps, camera_info, task_state, TASK_LABEL, TASK_FOCUS)


def build_force_choice_prompt(payload, camera_info=None, task_state=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, task_state, TASK_LABEL, TASK_FOCUS)


def score(payload, final_answer, camera_info=None, task_state=None):
    return common.score_for_task(payload, final_answer, camera_info, task_state, TASK_LABEL, TASK_FOCUS)
