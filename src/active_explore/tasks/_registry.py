from __future__ import annotations

from typing import Any


def _key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("&", "and").split())


LEGACY_TASK_MODULES = {
    "action": "tasks.action_sequencing.action_order_inference",
    "angle_confusion": "tasks.perceptual_grounding.view_hallucination",
    "cognitivemap": "tasks.cognitive_mapping.connectivity",
    "counting": "tasks.enumerative_perception.category_ambiguity",
    "deformable": "tasks.physical_structure.deformable",
    "distance": "tasks.metric_comparison.spatial_distance",
    "line": "tasks.spatial_relations.linear_alignment",
    "mirror": "tasks.specular_reflection.correspondence",
    "multiagent": "tasks.temporal_understanding.agent_observation",
    "occlusion": "tasks.perceptual_grounding.partial_occlusion",
    "pour": "tasks.physical_structure.liquid_volume",
    "size": "tasks.metric_comparison.dimensional_size",
    "slope": "tasks.physical_dynamics.inclined_plane",
    "stacking": "tasks.physical_dynamics.stacking_stability",
    "storage": "tasks.physical_structure.rigid_containment",
    "touching": "tasks.spatial_relations.physical_contact",
    "transparent": "tasks.perceptual_grounding.material_transparency",
    "triangle": "tasks.spatial_relations.geometric_configuration",
    "unobserved_changes": "tasks.temporal_understanding.unobserved_change",
}


SMALL_TASK_MODULES = {
    ("action sequencing", "action order inference"): "tasks.action_sequencing.action_order_inference",
    ("cognitive mapping", "connectivity"): "tasks.cognitive_mapping.connectivity",
    ("cognitive mapping", "long-term navigation"): "tasks.cognitive_mapping.long_term_navigation",
    ("cognitive mapping", "regional boundary"): "tasks.cognitive_mapping.regional_boundary",
    ("cognitive mapping", "traversable passage"): "tasks.cognitive_mapping.traversable_passage",
    ("enumerative perception", "category ambiguity"): "tasks.enumerative_perception.category_ambiguity",
    ("enumerative perception", "counting w occlusion"): "tasks.enumerative_perception.counting_w_occlusion",
    ("enumerative perception", "illumination variability"): "tasks.enumerative_perception.illumination_variability",
    ("enumerative perception", "merged observation"): "tasks.enumerative_perception.merged_observation",
    ("enumerative perception", "spatial segmentation"): "tasks.enumerative_perception.spatial_segmentation",
    ("enumerative perception", "structural enclosure"): "tasks.enumerative_perception.structural_enclosure",
    ("metric comparison", "dimensional size"): "tasks.metric_comparison.dimensional_size",
    ("metric comparison", "spatial distance"): "tasks.metric_comparison.spatial_distance",
    ("perceptual grounding", "material transparency"): "tasks.perceptual_grounding.material_transparency",
    ("perceptual grounding", "partial occlusion"): "tasks.perceptual_grounding.partial_occlusion",
    ("perceptual grounding", "view hallucination"): "tasks.perceptual_grounding.view_hallucination",
    ("physical dynamics", "inclined plane"): "tasks.physical_dynamics.inclined_plane",
    ("physical dynamics", "stacking and stability"): "tasks.physical_dynamics.stacking_stability",
    ("physical structure", "deformable"): "tasks.physical_structure.deformable",
    ("physical structure", "liquid volume"): "tasks.physical_structure.liquid_volume",
    ("physical structure", "rigid containment"): "tasks.physical_structure.rigid_containment",
    ("spatial relations", "geometric configuration"): "tasks.spatial_relations.geometric_configuration",
    ("spatial relations", "linear alignment"): "tasks.spatial_relations.linear_alignment",
    ("spatial relations", "physical contact"): "tasks.spatial_relations.physical_contact",
    ("specular reflection", "correspondence"): "tasks.specular_reflection.correspondence",
    ("specular reflection", "reflection authoring"): "tasks.specular_reflection.reflection_authoring",
    ("specular reflection", "spatial relations"): "tasks.specular_reflection.spatial_relations",
    ("temporal understanding", "agent observation"): "tasks.temporal_understanding.agent_observation",
    ("temporal understanding", "unobserved change"): "tasks.temporal_understanding.unobserved_change",
}


def module_for_payload(payload: dict[str, Any]) -> str | None:
    big_task = _key(payload.get("big_task") or payload.get("_hf_big_task"))
    small_task = _key(payload.get("small_task") or payload.get("_hf_small_task"))
    if not big_task or not small_task:
        return None
    return SMALL_TASK_MODULES.get((big_task, small_task))


def task_name_for_payload(payload: dict[str, Any]) -> str | None:
    module_name = module_for_payload(payload)
    if not module_name:
        return None
    return module_name.rsplit(".", 1)[-1]


def task_key_for_name(task_name: str) -> str:
    module_name = module_for_name(task_name)
    return module_name.rsplit(".", 1)[-1]


def module_for_name(task_name: str) -> str:
    normalized = str(task_name or "").strip()
    if "." in normalized:
        return normalized if normalized.startswith("tasks.") else f"tasks.{normalized}"
    if normalized in LEGACY_TASK_MODULES:
        return LEGACY_TASK_MODULES[normalized]
    for module_name in SMALL_TASK_MODULES.values():
        if module_name.rsplit(".", 1)[-1] == normalized:
            return module_name
    return f"tasks.{normalized}"
