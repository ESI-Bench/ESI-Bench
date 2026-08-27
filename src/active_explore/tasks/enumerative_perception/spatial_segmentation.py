from .common import *  # noqa: F401,F403
from . import common

TASK_NAME = "spatial_segmentation"
FULL_SCENE = True
TASK_LABEL = "Spatial Segmentation"
TASK_CASE = "observation_divided"
TASK_FOCUS = "Count targets distributed across divided spatial regions, keeping adjacent groups separated."
TASK_RULES = [
    "This case divides the observation across spatially separated regions.",
    "Combine counts across the separated regions as one scene-wide total.",
]


def build_system_prompt(payload, threshold, min_steps, camera_info=None):
    return common.build_system_prompt_for_task(
        payload, threshold, min_steps, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES
    )


def build_force_choice_prompt(payload, camera_info=None):
    return common.build_force_choice_prompt_for_task(payload, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)


def score(payload, final_answer, camera_info=None):
    return common.score_for_task(payload, final_answer, camera_info, TASK_LABEL, TASK_FOCUS, TASK_CASE, TASK_RULES)
