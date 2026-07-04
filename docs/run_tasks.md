# Active Task Names

Use the small-task name as `--task` when running `python src/main.py`.
`--metadata` may point to either a single question JSON or a big-task summary JSON under `dataset/json_clean`.

| Big task | Small task | `--task` |
|---|---|---|
| Action Sequencing | Action Order Inference | `action_order_inference` |
| Cognitive Mapping | Connectivity | `connectivity` |
| Cognitive Mapping | Long-Term Navigation | `long_term_navigation` |
| Cognitive Mapping | Regional Boundary | `regional_boundary` |
| Cognitive Mapping | Traversable Passage | `traversable_passage` |
| Enumerative Perception | Category Ambiguity | `category_ambiguity` |
| Enumerative Perception | Counting w Occlusion | `counting_w_occlusion` |
| Enumerative Perception | Illumination Variability | `illumination_variability` |
| Enumerative Perception | Merged Observation | `merged_observation` |
| Enumerative Perception | Spatial Segmentation | `spatial_segmentation` |
| Enumerative Perception | Structural Enclosure | `structural_enclosure` |
| Metric Comparison | Dimensional Size | `dimensional_size` |
| Metric Comparison | Spatial Distance | `spatial_distance` |
| Perceptual Grounding | Material Transparency | `material_transparency` |
| Perceptual Grounding | Partial Occlusion | `partial_occlusion` |
| Perceptual Grounding | View Hallucination | `view_hallucination` |
| Physical Dynamics | Inclined Plane | `inclined_plane` |
| Physical Dynamics | Stacking & Stability | `stacking_stability` |
| Physical Structure | Deformable | `deformable` |
| Physical Structure | Liquid Volume | `liquid_volume` |
| Physical Structure | Rigid Containment | `rigid_containment` |
| Spatial Relations | Geometric Configuration | `geometric_configuration` |
| Spatial Relations | Linear Alignment | `linear_alignment` |
| Spatial Relations | Physical Contact | `physical_contact` |
| Specular Reflection | Correspondence | `correspondence` |
| Specular Reflection | Reflection Authoring | `reflection_authoring` |
| Specular Reflection | Spatial Relations | `spatial_relations` |
| Temporal Understanding | Agent Observation | `agent_observation` |
| Temporal Understanding | Unobserved Change | `unobserved_change` |

Example:

```bash
python src/main.py \
  --task rigid_containment \
  --metadata "dataset/json_clean/Physical Structure/Rigid Containment/house_single_floor/kitchen_0/q_000.json" \
  --provider gemini \
  --model gemini-3.1-pro-preview \
  --results-root outputs/results \
  --step-image-root outputs/steps \
  --overwrite
```

Legacy names such as `counting`, `cognitivemap`, and `mirror` are kept as aliases for older automation, but new scripts should use the small-task names above.
