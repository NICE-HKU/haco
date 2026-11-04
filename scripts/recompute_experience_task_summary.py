#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def effective_network_latency(
    network_latency: Any,
    failure_type: Optional[str],
    network_packet_loss_timeout_sec: float,
    agent_failure_retry_penalty_sec: float,
) -> float:
    try:
        val = float(network_latency or 0.0)
    except Exception:
        val = 0.0

    if val == -1.0:
        return network_packet_loss_timeout_sec
    return max(val, 0.0)

def effective_agent_latency(
    agent_latency: Any,
    failure_type: Optional[str],
    agent_failure_retry_penalty_sec: float,
) -> float:
    try:
        val = float(agent_latency or 0.0)
    except Exception:
        val = 0.0

    if failure_type == "agent_failure_rate":
        return agent_failure_retry_penalty_sec
    return max(val, 0.0)


def infer_router_strategy_from_path(file_path: Path) -> Optional[str]:
    """
    Best-effort router strategy inference from the file path.
    """
    parts = [p.lower() for p in file_path.parts]
    joined = "/".join(parts)

    if "moa" in parts or "/moa/" in joined:
        return "moa"
    if any("cac" in p for p in parts):
        return "cac_router"
    if "bestone" in parts:
        return "bestone"
    if "random" in parts:
        return "random"
    return None


def calculate_task_agent_and_network_latency(
    task_records: List[Dict[str, Any]],
    router_strategy: Optional[str],
    network_packet_loss_timeout_sec: float,
    agent_failure_retry_penalty_sec: float,
) -> Tuple[float, float]:
    """
    Align with runtime summary semantics in experience_collecting.py:
    - moa: per step choose max(agent_latency + effective_network_latency)
    - others: per step use selected_agent attempts
    Returns (agent_latency_sum, network_latency_sum).
    """
    step_agent_records: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    step_router_records: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

    for record in task_records:
        step_key = (record.get("step_id"), record.get("small_step_id"))
        if record.get("agent_path") == "ROUTER_CALL":
            step_router_records[step_key] = record
        else:
            step_agent_records[step_key].append(record)

    if not step_router_records:
        agent_total = sum(
            effective_agent_latency(
                record.get("latency", 0.0),
                record.get("failure_type"),
                agent_failure_retry_penalty_sec,
            )
            for record in task_records
            if record.get("agent_path") != "ROUTER_CALL"
        )
        network_total = sum(
            effective_network_latency(
                record.get("network_latency", 0.0),
                record.get("failure_type"),
                network_packet_loss_timeout_sec,
                agent_failure_retry_penalty_sec,
            )
            for record in task_records
            if record.get("agent_path") != "ROUTER_CALL"
        )
        return agent_total, network_total

    total_agent_latency = 0.0
    total_network_latency = 0.0
    is_moa = router_strategy == "moa"

    for step_key, router_record in step_router_records.items():
        step_records = step_agent_records.get(step_key, [])
        if not step_records:
            continue

        if is_moa:
            selected_record = max(
                step_records,
                key=lambda record: effective_agent_latency(
                    record.get("latency", 0.0),
                    record.get("failure_type"),
                    agent_failure_retry_penalty_sec,
                )
                + effective_network_latency(
                    record.get("network_latency", 0.0),
                    record.get("failure_type"),
                    network_packet_loss_timeout_sec,
                    agent_failure_retry_penalty_sec,
                ),
            )
            total_agent_latency += effective_agent_latency(
                selected_record.get("latency", 0.0),
                selected_record.get("failure_type"),
                agent_failure_retry_penalty_sec,
            )
            total_network_latency += effective_network_latency(
                selected_record.get("network_latency", 0.0),
                selected_record.get("failure_type"),
                network_packet_loss_timeout_sec,
                agent_failure_retry_penalty_sec,
            )
            continue

        selected_agent = router_record.get("selected_agent")
        selected_records = [
            record for record in step_records if record.get("agent_path") == selected_agent
        ]
        records_to_count = selected_records if selected_records else step_records

        total_agent_latency += sum(
            effective_agent_latency(
                record.get("latency", 0.0),
                record.get("failure_type"),
                agent_failure_retry_penalty_sec,
            )
            for record in records_to_count
        )
        total_network_latency += sum(
            effective_network_latency(
                record.get("network_latency", 0.0),
                record.get("failure_type"),
                network_packet_loss_timeout_sec,
                agent_failure_retry_penalty_sec,
            )
            for record in records_to_count
        )

    return total_agent_latency, total_network_latency


def build_task_summary(
    session_id: str,
    task_id: str,
    task_records: List[Dict[str, Any]],
    old_end: Optional[Dict[str, Any]],
    router_strategy: Optional[str],
    network_packet_loss_timeout_sec: float,
    agent_failure_retry_penalty_sec: float,
) -> Dict[str, Any]:
    agent_total_latency = 0.0
    agent_total_tokens = 0
    router_total_latency = 0.0
    router_token_count = 0
    total_network_latency = 0.0

    call_count = 0
    step_num = 0
    network_failure_count = 0
    agent_failure_count = 0
    physical_failure_count = 0

    role_step_counts = defaultdict(int)
    role_agent_invocations = defaultdict(int)

    for record in task_records:
        if record.get("step_id") == "end":
            continue

        if record.get("agent_path") == "ROUTER_CALL":
            # Keep router aggregation consistent across strategies.
            step_num += 1
            router_total_latency += float(record.get("router_latency", 0.0) or 0.0)
            router_token_count += int(record.get("token_count", 0) or 0)

            role = record.get("role")
            if role:
                role_step_counts[role] += 1
        else:
            call_count += 1
            agent_total_tokens += int(record.get("token_count", 0) or 0)

            if record.get("network_latency") == -1.0:
                network_failure_count += 1

            if record.get("failure_type") == "agent_failure_rate":
                agent_failure_count += 1

            if record.get("failure_type") in ("network_packet_loss", "agent_failure_rate"):
                physical_failure_count += 1

            agent_path = record.get("agent_path")
            if isinstance(agent_path, str) and "-" in agent_path:
                role = agent_path.split("-", 1)[0]
                role_agent_invocations[role] += 1

    agent_total_latency, total_network_latency = calculate_task_agent_and_network_latency(
        task_records,
        router_strategy,
        network_packet_loss_timeout_sec,
        agent_failure_retry_penalty_sec,
    )

    network_failure_rate = network_failure_count / call_count if call_count > 0 else 0.0
    agent_failure_rate = agent_failure_count / call_count if call_count > 0 else 0.0
    physical_failure_attempt_rate = physical_failure_count / call_count if call_count > 0 else 0.0

    task_success = old_end.get("task_success") if old_end else None
    task_fail_reason = old_end.get("task_fail_reason") if old_end else None

    inferred_task_failed_due_physical = False
    if task_success is False:
        if task_fail_reason == "physical_failure":
            inferred_task_failed_due_physical = True
        elif call_count > 0 and physical_failure_count == call_count:
            inferred_task_failed_due_physical = True
    elif task_success is None:
        if call_count > 0 and physical_failure_count == call_count:
            inferred_task_failed_due_physical = True

    avg_role_call_counts = {}
    for role_name in ["planner", "coder", "debugger", "filter"]:
        invocations = role_agent_invocations[role_name]
        steps = role_step_counts[role_name]
        avg_role_call_counts[f"avg_{role_name}_call_count"] = invocations / steps if steps > 0 else 0

    total_latency = agent_total_latency + router_total_latency
    total_token_count = agent_total_tokens + router_token_count

    return {
        "session_id": str(session_id),
        "task_id": str(task_id),
        "step_id": "end",
        "agent_latency": agent_total_latency,
        "router_latency": router_total_latency,
        "network_latency": total_network_latency,
        "total_latency": total_latency,
        "agent_token_count": agent_total_tokens,
        "router_token_count": router_token_count,
        "total_token_count": total_token_count,
        "timestamp": time.time(),
        "call_count": call_count,
        "step_num": step_num,
        "network_failure_rate": network_failure_rate,
        "agent_failure_rate": agent_failure_rate,
        "physical_failure_count": physical_failure_count,
        "physical_failure_attempt_rate": physical_failure_attempt_rate,
        "task_success": task_success,
        "task_fail_reason": task_fail_reason,
        "task_failed_due_physical": inferred_task_failed_due_physical,
        "planner_step_num": role_step_counts["planner"],
        "coder_step_num": role_agent_invocations["coder"],
        "debug_step_num": role_agent_invocations["debugger"],
        "filter_step_num": role_agent_invocations["filter"],
        **avg_role_call_counts,
    }


def recompute_file(
    file_path: Path,
    in_place: bool,
    output_path: Optional[Path],
    backup_suffix: str,
    network_packet_loss_timeout_sec: float,
    agent_failure_retry_penalty_sec: float,
    router_strategy: Optional[str] = None,
    infer_router_strategy: bool = True,
) -> Path:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    raw_lines = file_path.read_text(encoding="utf-8").splitlines()

    parsed_lines: List[Dict[str, Any]] = []
    task_records = defaultdict(list)
    task_order: List[str] = []
    task_seen = set()
    old_end_by_task: Dict[str, Dict[str, Any]] = {}
    session_by_task: Dict[str, str] = {}

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        parsed_lines.append(rec)

        tid = str(rec.get("task_id")) if rec.get("task_id") is not None else None
        sid = str(rec.get("session_id")) if rec.get("session_id") is not None else ""
        if not tid:
            continue

        if tid not in task_seen:
            task_seen.add(tid)
            task_order.append(tid)

        if sid and tid not in session_by_task:
            session_by_task[tid] = sid

        if rec.get("step_id") == "end":
            old_end_by_task[tid] = rec
        else:
            task_records[tid].append(rec)

    router_strategy_for_file = router_strategy
    if router_strategy_for_file is None and infer_router_strategy:
        router_strategy_for_file = infer_router_strategy_from_path(file_path)

    new_end_by_task: Dict[str, Dict[str, Any]] = {}
    for tid in task_order:
        sid = session_by_task.get(tid, "")
        new_end_by_task[tid] = build_task_summary(
            session_id=sid,
            task_id=tid,
            task_records=task_records.get(tid, []),
            old_end=old_end_by_task.get(tid),
            router_strategy=router_strategy_for_file,
            network_packet_loss_timeout_sec=network_packet_loss_timeout_sec,
            agent_failure_retry_penalty_sec=agent_failure_retry_penalty_sec,
        )

    written_end_tasks = set()
    out_lines: List[str] = []

    for rec in parsed_lines:
        tid = str(rec.get("task_id")) if rec.get("task_id") is not None else None
        if rec.get("step_id") == "end":
            if tid and tid in new_end_by_task and tid not in written_end_tasks:
                out_lines.append(json.dumps(new_end_by_task[tid], ensure_ascii=False))
                written_end_tasks.add(tid)
            continue
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    # If some tasks had no existing end record, append one at file end.
    for tid in task_order:
        if tid not in written_end_tasks:
            out_lines.append(json.dumps(new_end_by_task[tid], ensure_ascii=False))

    if in_place:
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        shutil.copy2(file_path, backup_path)
        target = file_path
    else:
        target = output_path or file_path.with_suffix(file_path.suffix + ".recomputed")

    target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute step_id=end task summaries from existing experience_task.jsonl detail records."
    )
    parser.add_argument("input", type=Path, help="Path to experience_task.jsonl")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input file (creates backup).")
    parser.add_argument("--output", type=Path, default=None, help="Output file path (ignored if --in-place).")
    parser.add_argument("--backup-suffix", default=".bak_before_recompute", help="Backup suffix for --in-place mode.")
    parser.add_argument("--network-packet-loss-timeout-sec", type=float, default=10.0)
    parser.add_argument("--agent-failure-retry-penalty-sec", type=float, default=2.0)
    parser.add_argument(
        "--router-strategy",
        choices=["moa", "cac_router", "bestone", "random"],
        default=None,
        help="Explicit router strategy. If omitted, strategy can be inferred from file path.",
    )
    parser.add_argument(
        "--no-infer-router-strategy",
        action="store_true",
        help="Disable path-based router strategy inference when --router-strategy is not provided.",
    )

    args = parser.parse_args()

    target = recompute_file(
        file_path=args.input,
        in_place=args.in_place,
        output_path=args.output,
        backup_suffix=args.backup_suffix,
        network_packet_loss_timeout_sec=args.network_packet_loss_timeout_sec,
        agent_failure_retry_penalty_sec=args.agent_failure_retry_penalty_sec,
        router_strategy=args.router_strategy,
        infer_router_strategy=not args.no_infer_router_strategy,
    )
    print(f"Recomputed summaries written to: {target}")
    if args.in_place:
        print(f"Backup file created with suffix: {args.backup_suffix}")


if __name__ == "__main__":
    main()
