from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from passive_gt.manifest import AVAILABLE_VIEW_STATUSES
from passive_gt.pipeline import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_EXISTING_ROOTS,
    DEFAULT_OUTPUT_ROOT,
    ExistingImageResolver,
    build_records,
    materialize_records,
    write_root_manifests,
)
from passive_gt.replay import SUPPORTED_RENDER_MODES
from passive_gt.regeneration import render_or_regenerate_sample


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and materialize Passive-GT views from ESI-Bench json_clean metadata."
    )
    parser.add_argument("command", choices=("plan", "collect", "render", "_render-one"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--existing-root", action="append", type=Path, default=[])
    parser.add_argument("--big-task", default="")
    parser.add_argument("--small-task", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--source-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--only-view-id", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--unobserved-container-state",
        choices=("reveal", "closed"),
        default="reveal",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--conda-env", default="behavior")
    parser.add_argument("--headless", action="store_true", help="Use Isaac headless mode (GUI mode is more stable on this install).")
    parser.add_argument("--worker-timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def merge_with_root_manifest(output_root: Path, selected: list[dict]) -> list[dict]:
    """Keep unselected records when a filtered command refreshes root metadata."""
    path = output_root / "manifest.jsonl"
    if not path.is_file():
        return selected
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(record.get("source_json") or "")
            if key:
                records[key] = record
    for record in selected:
        records[str(record.get("source_json") or "")] = record
    return [records[key] for key in sorted(records)]


def cleanup_dead_rstring_shm() -> list[str]:
    """Remove only per-PID Carbonite files whose owner process is gone."""
    removed = []
    shm = Path("/dev/shm")
    for pattern in ("carb-RStringInternals-*", "sem.carb-RStringInternals-*"):
        for path in shm.glob(pattern):
            try:
                pid = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            try:
                os.kill(pid, 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                pass
    return removed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "_render-one":
        if args.source_json is None:
            raise SystemExit("_render-one requires --source-json")
        result = render_or_regenerate_sample(
            args.source_json.resolve(),
            dataset_root,
            output_root,
            overwrite=args.overwrite,
            only_view_ids=set(args.only_view_id) or None,
            unobserved_container_state=args.unobserved_container_state,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 1 if result.get("failed") else 0

    records = build_records(
        dataset_root,
        output_root,
        big_task=args.big_task or None,
        small_task=args.small_task or None,
        limit=max(args.limit, 0),
    )
    if args.command == "collect":
        roots = tuple(args.existing_root) if args.existing_root else DEFAULT_EXISTING_ROOTS
        resolver = ExistingImageResolver(roots)
        materialize_records(records, dataset_root, output_root, resolver, link_mode=args.link_mode)
    elif args.command == "render":
        candidates = []
        for record in records:
            if any(
                (
                    view.get("render_mode") in SUPPORTED_RENDER_MODES
                    or (
                        record.get("small_task") == "Liquid Volume"
                        and view.get("render_mode") == "full_regeneration"
                    )
                    or (
                        record.get("small_task") == "Inclined Plane"
                        and view.get("render_mode") == "dynamic_replay"
                    )
                    or (
                        record.get("small_task") == "Agent Observation"
                        and view.get("render_mode") == "dynamic_replay"
                    )
                    or (
                        record.get("small_task") == "Unobserved Change"
                        and view.get("render_mode") == "state_snapshot"
                    )
                )
                and (args.overwrite or view.get("status") not in AVAILABLE_VIEW_STATUSES)
                for view in (record.get("views") or [])
            ):
                candidates.append(record)
        worker_results = []
        for index, record in enumerate(candidates, start=1):
            source = dataset_root / record["source_json"]
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                args.conda_env,
                "python",
                str(Path(__file__).resolve()),
                "_render-one",
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
                "--source-json",
                str(source),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            if args.headless:
                env["OMNIGIBSON_HEADLESS"] = "True"
            else:
                env.setdefault("OMNIGIBSON_HEADLESS", "False")
            env.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "True")
            env.setdefault("OG_DISABLE_EMITTER_APIS", "1")
            env.setdefault("PYTHONFAULTHANDLER", "1")
            env.setdefault("MALLOC_ARENA_MAX", "2")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env.setdefault("PXR_WORK_THREAD_LIMIT", "2")
            print(json.dumps({"worker": index, "total": len(candidates), "source_json": str(source)}, ensure_ascii=False), flush=True)
            attempts = []
            for attempt in range(1, max(args.max_retries, 1) + 1):
                removed_shm = cleanup_dead_rstring_shm()
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=Path(__file__).resolve().parents[1],
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=args.worker_timeout,
                    )
                    attempt_result = {
                        "attempt": attempt,
                        "returncode": completed.returncode,
                        "stdout_tail": completed.stdout[-4000:],
                        "stderr_tail": completed.stderr[-4000:],
                        "removed_stale_shm": removed_shm,
                    }
                except subprocess.TimeoutExpired as exc:
                    attempt_result = {
                        "attempt": attempt,
                        "returncode": -1,
                        "error": "worker_timeout",
                        "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                        "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                        "removed_stale_shm": removed_shm,
                    }
                attempts.append(attempt_result)
                if attempt_result["returncode"] == 0:
                    break
                print(json.dumps({"source_json": str(source), **attempt_result}, ensure_ascii=False), flush=True)
            result = {"source_json": str(source), **attempts[-1], "attempts": attempts}
            worker_results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        records = build_records(
            dataset_root,
            output_root,
            big_task=args.big_task or None,
            small_task=args.small_task or None,
            limit=max(args.limit, 0),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "render_workers.json").open("w", encoding="utf-8") as stream:
            json.dump(worker_results, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if args.big_task or args.small_task or args.limit:
        records = merge_with_root_manifest(output_root, records)
    summary = write_root_manifests(output_root, records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
