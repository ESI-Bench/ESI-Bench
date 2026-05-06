"""
aggregate_fillable_results.py

Reads all result.json files under a batch_water directory,
filters success=True and particles_in_container > 20,
outputs a sorted JSON of category, model, and capacity.

Usage:
    python aggregate_fillable_results.py --input_root batch_water --output fillable_capacity.json
"""

import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="batch_water")
    parser.add_argument("--output",     type=str, default="fillable_capacity.json")
    args = parser.parse_args()

    results = []

    for folder in os.listdir(args.input_root):
        folder_path = os.path.join(args.input_root, folder)
        result_path = os.path.join(folder_path, "result.json")

        if not os.path.isdir(folder_path) or not os.path.exists(result_path):
            continue

        with open(result_path) as f:
            try:
                r = json.load(f)
            except Exception as e:
                print(f"[warn] Could not parse {result_path}: {e}")
                continue

        if not r.get("success", False):
            continue
        if r.get("particles_in_container", 0) <= 20:
            continue

        results.append({
            "category":             r["category"],
            "model":                r["model"],
            "particles_in_container": r["particles_in_container"],
        })

    results.sort(key=lambda x: x["particles_in_container"])

    output = {
        "total": len(results),
        "instances": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Found {len(results)} valid instances -> {args.output}")
    print(f"  Smallest: {results[0]['category']} / {results[0]['model']} = {results[0]['particles_in_container']}")
    print(f"  Largest:  {results[-1]['category']} / {results[-1]['model']} = {results[-1]['particles_in_container']}")


if __name__ == "__main__":
    main()