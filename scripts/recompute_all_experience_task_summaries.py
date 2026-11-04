#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional

from recompute_experience_task_summary import recompute_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-recompute step_id=end summaries for all experience_task.jsonl files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("evaluation/experimental_results"),
        help="Root directory to search for experience_task.jsonl files.",
    )
    parser.add_argument(
        "--glob",
        default="**/experience_task.jsonl",
        help="Glob pattern under --root for target files.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite input files (creates backups).")
    parser.add_argument(
        "--backup-suffix",
        default=".bak_before_recompute",
        help="Backup suffix used in --in-place mode.",
    )
    parser.add_argument("--network-packet-loss-timeout-sec", type=float, default=10.0)
    parser.add_argument("--agent-failure-retry-penalty-sec", type=float, default=2.0)
    parser.add_argument(
        "--router-strategy",
        choices=["moa", "cac_router", "bestone", "random"],
        default=None,
        help="Explicit router strategy for all files. If omitted, infer from path per file.",
    )
    parser.add_argument(
        "--no-infer-router-strategy",
        action="store_true",
        help="Disable path-based router strategy inference when --router-strategy is not provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched files; do not write outputs.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(root.glob(args.glob))
    print(f"Found {len(files)} files under {root} matching {args.glob}")

    if args.dry_run:
        for fp in files:
            print(f"[DRY-RUN] {fp}")
        return

    updated = 0
    failed = 0
    for fp in files:
        try:
            target = recompute_file(
                file_path=fp,
                in_place=args.in_place,
                output_path=None,
                backup_suffix=args.backup_suffix,
                network_packet_loss_timeout_sec=args.network_packet_loss_timeout_sec,
                agent_failure_retry_penalty_sec=args.agent_failure_retry_penalty_sec,
                router_strategy=args.router_strategy,
                infer_router_strategy=not args.no_infer_router_strategy,
            )
            updated += 1
            print(f"[OK] {fp} -> {target}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {fp}: {exc}")

    print(f"Done. updated={updated}, failed={failed}")


if __name__ == "__main__":
    main()

