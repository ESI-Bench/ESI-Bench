"""
build_fillable_manifest.py

Scans object_inventory.json (providers keys formatted as "category-model_id")
for every model of each fillable category and writes fillable_manifest.json:

{
  "bowl": ["model_id_1", "model_id_2", ...],
  "mug":  ["model_id_a"],
  ...
}

Run this once before the batch bash script.
"""

import json
import os
import sys

FILLABLE_CATEGORIES = [
    "bowl", "bucket", "carafe", "cruet", "kettle", "mug", "teapot",
    "colander", "coffee_cup", "tray", "jug", "canister", "wok",
    "paper_cup", "shopping_basket", "crock_pot",
    "wineglass", "vase", "soda_cup", "water_glass", "beer_glass",
    "reed_diffuser", "round_bottom_flask", "graduated_cylinder",
    "erlenmeyer_flask", "stockpot", "specimen_bottle", "pencil_holder",
    "pill_bottle", "toy_box", "storage_box", "reagent_bottle",
    "ice_bucket", "china", "goblet", "litter_box", "plant_pot",
    "steamer_basket", "teacup", "cocktail_glass", "drip_pot",
    "bucket_of_paint", "jar_of_cumin", "lunch_box", "packing_box",
    "ice_tray", "rum_bottle", "beer_bottle", "ink_bottle",
    "detergent_bottle", "hydrogen_peroxide_bottle", "saddle_soap_bottle",
    "pineapple_juice_carton", "jar_of_kidney_beans", "jar_of_clove",
    "jar_of_peppercorns", "hingeless_jar",
]

INVENTORY_PATHS = [
    "bddl3/bddl/generated_data/object_inventory.json",
    os.path.join(os.path.dirname(__file__), "object_inventory.json"),
]

OUTPUT_PATH = "fillable_manifest.json"


def load_inventory() -> dict:
    for path in INVENTORY_PATHS:
        if os.path.exists(path):
            print(f"[inventory] Loaded from: {path}")
            with open(path) as f:
                return json.load(f)
    raise RuntimeError(
        "No object_inventory.json found. Tried:\n" +
        "\n".join(f"  {p}" for p in INVENTORY_PATHS)
    )


def get_models_for_category(category: str, inventory: dict) -> list:
    """Return all model IDs for a category using providers key format 'category-model_id'."""
    providers = inventory.get("providers", inventory)
    matches   = [k for k in providers if k.startswith(f"{category}-")]
    return [k.split("-", 1)[1] for k in matches]


def main():
    inventory = load_inventory()

    manifest = {}
    missing  = []

    for cat in FILLABLE_CATEGORIES:
        cat_lower = cat.lower()
        models = get_models_for_category(cat_lower, inventory)
        if models:
            manifest[cat_lower] = models
            print(f"  [{cat_lower}]  {len(models)} model(s): {models}")
        else:
            print(f"  [{cat_lower}]  NO models found")
            missing.append(cat_lower)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    total_instances = sum(len(v) for v in manifest.values())
    print(f"\nManifest written to: {OUTPUT_PATH}")
    print(f"  Categories with models : {len(manifest)}")
    print(f"  Categories missing     : {len(missing)}")
    if missing:
        print(f"  Missing               : {missing}")
    print(f"  Total instances        : {total_instances}")


if __name__ == "__main__":
    main()