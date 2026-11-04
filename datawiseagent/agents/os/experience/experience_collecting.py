import json
import time
from collections import defaultdict
import os
import asyncio
import aiofiles
import math
from typing import Optional

from datawiseagent.common.config import global_config

_network_log_lock = asyncio.Lock()

def _is_haco_wo_experience_harvesting_active() -> bool:
    """Whether the current router strategy is HACO w/o experience harvesting."""
    strategy = global_config.get("agent", {}).get("selection_strategy", "")
    return strategy == "haco_wo_experience_harvesting"

def _get_experience_dir() -> str:
    """Dynamically retrieves the experience directory from global_config."""
    # This ensures that the directory is always read after global_config has been fully populated.
    return global_config.get("log", {}).get("experience_dir", "datawiseagent/agents/os/experience/")

def _get_task_log_path() -> str:
    """Dynamically constructs the path for the task log file."""
    return os.path.join(_get_experience_dir(), "experience_task.jsonl")

def _get_agent_log_path() -> str:
    """Dynamically constructs the path for the agent log file."""
    return os.path.join(_get_experience_dir(), "experience_agent.jsonl")

def _get_network_log_path() -> str:
    """Dynamically constructs the path for the network log file."""
    return os.path.join(_get_experience_dir(), "experience_network.jsonl")

def _get_network_packet_loss_timeout_sec() -> float:
    """
    Gets timeout penalty (seconds) for network packet loss attempts.
    Backward-compatible: falls back to network.timeout_penalty_sec if specific key is absent.
    """
    network_cfg = global_config.get("network", {})
    return float(
        network_cfg.get(
            "network_packet_loss_timeout_sec",
            network_cfg.get("timeout_penalty_sec", 10.0),
        )
    )

def _get_agent_failure_retry_penalty_sec() -> float:
    """
    Gets retry penalty (seconds) for simulated agent failure attempts.
    Backward-compatible default is lower than network timeout.
    """
    network_cfg = global_config.get("network", {})
    return float(network_cfg.get("agent_failure_retry_penalty_sec", 2.0))

def _effective_agent_latency(latency: float, failure_type: Optional[str] = None) -> float:
    """
    Converts raw agent latency to an effective value for cost aggregation.
    Agent failure retry penalty is accounted as agent-side overhead.
    """
    if failure_type == "agent_failure_rate":
        return _get_agent_failure_retry_penalty_sec()
    return max(float(latency or 0.0), 0.0)

def _effective_network_latency(network_latency: float, failure_type: Optional[str] = None) -> float:
    """
    Converts raw network latency to an effective value for cost aggregation.
    Keep raw -1.0 in logs for loss-rate modeling; only convert during summary.
    """
    if network_latency == -1.0:
        return _get_network_packet_loss_timeout_sec()
    return max(float(network_latency or 0.0), 0.0)


def _calculate_task_network_latency(task_records: list, router_strategy: Optional[str]) -> float:
    """
    Calculates task-level network latency using router-specific semantics.
    - moa: per step, take the max effective network latency across all attempts.
    - others: per step, sum effective network latency of the selected agent's attempts.
    If selected agent cannot be resolved, fallback to summing all attempts of that step.
    """
    step_agent_records = defaultdict(list)
    step_router_records = {}

    for record in task_records:
        if record.get("agent_path") == "ROUTER_CALL":
            step_key = (record.get("step_id"), record.get("small_step_id"))
            step_router_records[step_key] = record
        else:
            step_key = (record.get("step_id"), record.get("small_step_id"))
            step_agent_records[step_key].append(record)

    # Fallback for logs without router records.
    if not step_router_records:
        return sum(
            _effective_network_latency(
                record.get("network_latency", 0),
                record.get("failure_type"),
            )
            for record in task_records
            if record.get("agent_path") != "ROUTER_CALL"
        )

    total_network_latency = 0.0
    is_moa = router_strategy == "moa"

    for step_key, router_record in step_router_records.items():
        step_records = step_agent_records.get(step_key, [])
        if not step_records:
            continue

        if is_moa:
            total_network_latency += max(
                _effective_network_latency(
                    record.get("network_latency", 0),
                    record.get("failure_type"),
                )
                for record in step_records
            )
            continue

        selected_agent = router_record.get("selected_agent")
        selected_records = [
            record for record in step_records if record.get("agent_path") == selected_agent
        ]
        records_to_count = selected_records if selected_records else step_records

        total_network_latency += sum(
            _effective_network_latency(
                record.get("network_latency", 0),
                record.get("failure_type"),
            )
            for record in records_to_count
        )

    return total_network_latency


def _calculate_task_agent_and_network_latency(task_records: list, router_strategy: Optional[str]) -> tuple[float, float]:
    """
    Calculates task-level agent/network latency with router-specific semantics.
    - moa: per step select the max (agent_latency + effective_network_latency) attempt.
    - others: per step aggregate attempts of the selected agent.
    Returns: (agent_latency_sum, effective_network_latency_sum).
    """
    step_agent_records = defaultdict(list)
    step_router_records = {}

    for record in task_records:
        if record.get("agent_path") == "ROUTER_CALL":
            step_key = (record.get("step_id"), record.get("small_step_id"))
            step_router_records[step_key] = record
        else:
            step_key = (record.get("step_id"), record.get("small_step_id"))
            step_agent_records[step_key].append(record)

    # Fallback for logs without router records.
    if not step_router_records:
        agent_total = sum(
            _effective_agent_latency(
                record.get("latency", 0.0),
                record.get("failure_type"),
            )
            for record in task_records
            if record.get("agent_path") != "ROUTER_CALL"
        )
        network_total = _calculate_task_network_latency(task_records, router_strategy)
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
                key=lambda record: _effective_agent_latency(
                    record.get("latency", 0.0),
                    record.get("failure_type"),
                )
                + _effective_network_latency(
                    record.get("network_latency", 0),
                    record.get("failure_type"),
                ),
            )
            total_agent_latency += _effective_agent_latency(
                selected_record.get("latency", 0.0),
                selected_record.get("failure_type"),
            )
            total_network_latency += _effective_network_latency(
                selected_record.get("network_latency", 0),
                selected_record.get("failure_type"),
            )
            continue

        selected_agent = router_record.get("selected_agent")
        selected_records = [
            record for record in step_records if record.get("agent_path") == selected_agent
        ]
        records_to_count = selected_records if selected_records else step_records

        total_agent_latency += sum(
            _effective_agent_latency(
                record.get("latency", 0.0),
                record.get("failure_type"),
            )
            for record in records_to_count
        )
        total_network_latency += sum(
            _effective_network_latency(
                record.get("network_latency", 0),
                record.get("failure_type"),
            )
            for record in records_to_count
        )

    return total_agent_latency, total_network_latency

async def update_experience_network(
    agent_from: str,
    agent_to: str,
    task_id: str,
    network_latency: float,
    cell_size: int,
    network_mode: str = "simulated",
): # Type hint reverted
    """记录并更新 agent 之间的网络交互图,并估算带宽和丢包率。"""
    if _is_haco_wo_experience_harvesting_active():
        return

    if agent_from is None:
        agent_from = "router"
        
    network_log_path = _get_network_log_path()
    
    async with _network_log_lock:
        graph = {}
        if os.path.exists(network_log_path) and os.path.getsize(network_log_path) > 0:
            async with aiofiles.open(network_log_path, "r", encoding="utf-8") as f:
                try:
                    content = await f.read()
                    graph = json.loads(content)
                except json.JSONDecodeError:
                    graph = {}

        u, v = sorted([agent_from, agent_to])

        interaction = {
            "task_id": task_id,
            "network_latency": network_latency,
            "cell_size": cell_size,
            "network_mode": network_mode,
            "timestamp": time.time(),
        }

        # Get or create the link data structure
        if u not in graph:
            graph[u] = {}
        if v not in graph[u]:
            graph[u][v] = {"interactions": []}

        # Compatibility with old list-based format
        if isinstance(graph[u][v], list):
            graph[u][v] = {"interactions": graph[u][v]}

        graph[u][v]["interactions"].append(interaction)

        # --- Calculate mode-aware stats ---
        interactions = graph[u][v]["interactions"]
        simulated_interactions = [
            item for item in interactions
            if item.get("network_mode", "simulated") != "real_probe"
        ]
        real_interactions = [
            item for item in interactions
            if item.get("network_mode") == "real_probe"
        ]

        # Simulated mode stats (backward-compatible keys)
        bandwidths = []
        failed_attempts = 0
        total_attempts = len(simulated_interactions)
        for item in simulated_interactions:
            if item.get("network_latency", 0) > 0:
                bandwidths.append(item["cell_size"] / item["network_latency"])
            elif item.get("network_latency") == -1.0:
                failed_attempts += 1

        loss_rate = failed_attempts / total_attempts if total_attempts > 0 else 0.0
        if bandwidths:
            avg_bandwidth = sum(bandwidths) / len(bandwidths)
            if len(bandwidths) > 1:
                variance = sum([(x - avg_bandwidth) ** 2 for x in bandwidths]) / len(bandwidths)
                bandwidth_jitter = math.sqrt(variance)
            else:
                bandwidth_jitter = 0
        else:
            avg_bandwidth = 0
            bandwidth_jitter = 0

        graph[u][v]["avg_bandwidth"] = avg_bandwidth
        graph[u][v]["bandwidth_jitter"] = bandwidth_jitter
        graph[u][v]["loss_rate"] = loss_rate

        # Real probe mode stats (RTT-focused)
        real_rtts = [
            float(item.get("network_latency", 0) or 0)
            for item in real_interactions
            if float(item.get("network_latency", 0) or 0) > 0
        ]
        real_total = len(real_interactions)
        real_failures = real_total - len(real_rtts)
        if real_rtts:
            avg_rtt = sum(real_rtts) / len(real_rtts)
            if len(real_rtts) > 1:
                variance = sum([(x - avg_rtt) ** 2 for x in real_rtts]) / len(real_rtts)
                rtt_jitter = math.sqrt(variance)
            else:
                rtt_jitter = 0.0
        else:
            avg_rtt = 0.0
            rtt_jitter = 0.0
        probe_failure_rate = real_failures / real_total if real_total > 0 else 0.0

        graph[u][v]["avg_rtt"] = avg_rtt
        graph[u][v]["rtt_jitter"] = rtt_jitter
        graph[u][v]["probe_failure_rate"] = probe_failure_rate


        async with aiofiles.open(network_log_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(graph, ensure_ascii=False, indent=4))

def log_task_experience(session_id: str, task_id: str, step_id: int, small_step_id: int, agent_path: str, latency: float, token_count: int, network_latency: float = 0.0, cell_size: int = 0, cac_data: dict = None, failure_type: Optional[str] = None):
    """记录单次子代理调用的详细信息。"""
    record = {
        "session_id": str(session_id),
        "task_id": str(task_id),
        "step_id": step_id,
        "small_step_id": small_step_id,
        "agent_path": agent_path,
        "latency": latency,
        "token_count": token_count,
        "network_latency": network_latency,
        "cell_size": cell_size,
        "timestamp": time.time(),
    }
    if cac_data:
        record["cac"] = cac_data
    if failure_type:
        record["failure_type"] = failure_type
    with open(_get_task_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_router_experience(
    session_id: str,
    task_id: str,
    step_id: int,
    small_step_id: int,
    router_latency: float,
    selected_agent: str = None,
    agent_scores: list = None,
    router_token_count: int = 0,
    network_latency: float = 0.0,
    role: str = None,
    call_count: int = 0,
    selection_latency: Optional[float] = None,
    aggregation_latency: Optional[float] = None,
    failure_type: Optional[str] = None,
): # Type hint reverted
    """记录单次路由器调用（选择+聚合）的详细信息。"""
    total_latency = router_latency + network_latency # Logic reverted
    record = {
        "session_id": str(session_id),
        "task_id": str(task_id),
        "step_id": step_id,
        "small_step_id": small_step_id,
        "agent_path": "ROUTER_CALL",
        "total_latency": total_latency,
        "token_count": router_token_count,
        "router_latency": router_latency,
        "network_latency": network_latency,
        "selected_agent": selected_agent,
        "agent_scores": agent_scores,
        "timestamp": time.time(),
        "role": role,
        "call_count": call_count,
    }
    if selection_latency is not None:
        record["selection_latency"] = selection_latency
    if aggregation_latency is not None:
        record["aggregation_latency"] = aggregation_latency
    if failure_type:
        record["failure_type"] = failure_type
    with open(_get_task_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def update_agent_experience():
    """在任务结束后，根据 task log 重新计算并更新 agent 的聚合统计数据。"""
    if _is_haco_wo_experience_harvesting_active():
        return

    task_log_file = _get_task_log_path()
    agent_log_file = _get_agent_log_path()

    if not os.path.exists(task_log_file):
        return

    agent_stats = defaultdict(
        lambda: {
            "total_latency": 0,
            "total_tokens": 0,
            "call_count": 0,
            "success_total_latency": 0,
            "success_total_tokens": 0,
            "success_call_count": 0,
            "network_failure_count": 0,
            "agent_failure_count": 0,
            "json_parse_failure_count": 0,
        }
    )
    agent_scores_history = defaultdict(list) # New: To store scores for each agent
    router_step_keys = set()
    router_scored_agents_by_step = defaultdict(set)

    with open(task_log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if not record.get("agent_path") or record.get("step_id") == "end":
                    continue
                if record.get("agent_path") == "ROUTER_CALL":
                    step_key = (str(record.get("task_id")), record.get("step_id"), record.get("small_step_id"))
                    router_step_keys.add(step_key)
                    agent_scores = record.get("agent_scores") or []
                    if isinstance(agent_scores, list):
                        for agent_score_item in agent_scores:
                            scored_agent_name = agent_score_item.get("agent_name")
                            score = agent_score_item.get("score")
                            if scored_agent_name and isinstance(score, (int, float)):
                                router_scored_agents_by_step[step_key].add(scored_agent_name)
            except (json.JSONDecodeError, KeyError):
                continue

    scored_success_call_count = defaultdict(int)

    with open(task_log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                
                agent_path = record.get("agent_path")
                if not agent_path or record.get("step_id") == "end":
                    continue
                
                if agent_path == "ROUTER_CALL":
                    agent_scores = record.get("agent_scores")
                    router_failure_type = record.get("failure_type")
                    if agent_scores:
                        for agent_score_item in agent_scores:
                            scored_agent_name = agent_score_item.get("agent_name")
                            score = agent_score_item.get("score")
                            if scored_agent_name and score is not None:
                                # Normalize the score from 1-10 to 0-1
                                agent_scores_history[scored_agent_name].append(score / 10.0)
                            if (
                                router_failure_type == "json_parse_failed"
                                and scored_agent_name
                                and agent_score_item.get("reasoning") == "parse_failed"
                            ):
                                agent_stats[scored_agent_name]["json_parse_failure_count"] += 1
                else:
                    agent_stats[agent_path]["total_latency"] += record.get("latency", 0)
                    agent_stats[agent_path]["total_tokens"] += record.get("token_count", 0)
                    agent_stats[agent_path]["call_count"] += 1
                    if not record.get("failure_type"):
                        agent_stats[agent_path]["success_total_latency"] += record.get("latency", 0)
                        agent_stats[agent_path]["success_total_tokens"] += record.get("token_count", 0)
                        agent_stats[agent_path]["success_call_count"] += 1

                        step_key = (str(record.get("task_id")), record.get("step_id"), record.get("small_step_id"))
                        if (
                            step_key in router_step_keys
                            and agent_path in router_scored_agents_by_step.get(step_key, set())
                        ):
                            scored_success_call_count[agent_path] += 1
                    if record.get("failure_type") == "network_packet_loss":
                        agent_stats[agent_path]["network_failure_count"] += 1
                    if record.get("failure_type") == "agent_failure_rate":
                        agent_stats[agent_path]["agent_failure_count"] += 1
            except (json.JSONDecodeError, KeyError):
                continue

    updated_records = []
    # Merge agent_stats and agent_scores_history
    all_agents = set(agent_stats.keys()) | set(agent_scores_history.keys())

    for agent_path in all_agents:
        stats = agent_stats[agent_path]
        scores = agent_scores_history[agent_path]

        count = stats["call_count"]
        network_failure_count = stats["network_failure_count"]
        agent_failure_count = stats["agent_failure_count"]
        json_parse_failure_count = stats["json_parse_failure_count"]
        
        success_count = stats["success_call_count"]
        scored_success_count = scored_success_call_count[agent_path]
        avg_latency = stats["success_total_latency"] / success_count if success_count > 0 else 0
        avg_token_count = stats["success_total_tokens"] / success_count if success_count > 0 else 0
        # Average score should only use successful calls from steps that reached router evaluation.
        avg_score = sum(scores) / scored_success_count if scored_success_count > 0 else 0
        
        agent_failure_rate = agent_failure_count / count if count > 0 else 0.0

        # Keep Beta parameters aligned with HACO router's UCB calculation.
        alpha = (avg_score * count) + 1.0
        beta = ((1.0 - avg_score) * count) + 1.0
        beta_sum = alpha + beta
        mean = alpha / beta_sum if beta_sum > 0 else 0.5
        variance = (alpha * beta) / ((beta_sum ** 2) * (beta_sum + 1)) if beta_sum > 0 else 0.0
        std = math.sqrt(variance)

        record = {
            "agent_path": agent_path,
            "avg_latency": avg_latency,
            "avg_token_count": avg_token_count,
            "call_count": count,
            "network_failure_count": network_failure_count,
            "success_call_count": success_count,
            "scored_success_call_count": scored_success_count,
            "avg_score": avg_score, # New: Add avg_score to the record
            "agent_failure_count": agent_failure_count,
            "agent_failure_rate": agent_failure_rate,
            "json_parse_failure_count": json_parse_failure_count,
            "alpha": alpha,
            "beta": beta,
            "mean": mean,
            "std": std,
        }
        updated_records.append(record)

    with open(agent_log_file, "w", encoding="utf-8") as f:
        for record in updated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def calculate_and_log_task_summary(
    session_id: str,
    task_id: str,
    task_success: Optional[bool] = None,
    task_fail_reason: Optional[str] = None,
    router_strategy: Optional[str] = None,
):
    """
    计算并记录单个任务的汇总信息。
    """
    task_log_file = _get_task_log_path()
    
    if not os.path.exists(task_log_file):
        return

    agent_total_latency = 0
    agent_total_tokens = 0
    router_total_latency = 0
    router_token_count = 0
    total_network_latency = 0.0

    # New metrics for task summary
    call_count = 0
    step_num = 0
    network_failure_count = 0
    agent_failure_count = 0 # New: Count of agent failures due to failure_rate
    physical_failure_count = 0
    
    role_step_counts = defaultdict(int) # e.g., {'planner': 5, 'coder': 10}
    role_agent_invocations = defaultdict(int) # e.g., {'planner': 5, 'coder': 10}
    # Decision-stage call counts from router logs; excludes retry attempts by design.
    role_decision_call_counts = defaultdict(float)
    
    target_task_id = str(task_id)
    
    task_records = []
    with open(task_log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if str(record.get("task_id")) == target_task_id and record.get("step_id") != "end":
                    task_records.append(record)
            except json.JSONDecodeError:
                continue

    # First pass to calculate various counts and sums
    for record in task_records:
        if record.get("step_id") == "end":
            continue

        if record.get("agent_path") == "ROUTER_CALL":
            router_token_count_val = record.get("token_count", 0)

            # Keep router aggregation consistent across strategies:
            # random often has token_count=0 (no evaluator LLM), but
            # its router call still represents a real routing step/latency.
            step_num += 1 # Count total router steps
            router_total_latency += record.get("router_latency", 0)
            router_token_count += router_token_count_val
            
            # Count role-specific router steps
            role = record.get("role")
            if role:
                role_step_counts[role] += 1
                role_decision_call_counts[role] += float(record.get("call_count", 0) or 0)
            
        else: # This is a regular agent invocation record
            call_count += 1 # Count total agent invocations
            agent_total_latency += record.get("latency", 0)
            agent_total_tokens += record.get("token_count", 0)
            # Count network failures
            if record.get("network_latency") == -1.0:
                network_failure_count += 1
            
            # Count agent failures due to failure_rate
            if record.get("failure_type") == "agent_failure_rate":
                agent_failure_count += 1

            if record.get("failure_type") in ("network_packet_loss", "agent_failure_rate"):
                physical_failure_count += 1
            
            # Count role-specific agent invocations
            agent_path = record.get("agent_path")
            if agent_path:
                role = agent_path.split('-')[0] # Assuming format like "role-agentname"
                role_agent_invocations[role] += 1

    agent_total_latency, total_network_latency = _calculate_task_agent_and_network_latency(
        task_records, router_strategy
    )

    # Calculate network_failure_rate
    network_failure_rate = network_failure_count / call_count if call_count > 0 else 0.0
    # Calculate agent_failure_rate
    agent_failure_rate = agent_failure_count / call_count if call_count > 0 else 0.0
    physical_failure_attempt_rate = physical_failure_count / call_count if call_count > 0 else 0.0

    inferred_task_failed_due_physical = False
    if task_success is False:
        if task_fail_reason == "physical_failure":
            inferred_task_failed_due_physical = True
        elif call_count > 0 and physical_failure_count == call_count:
            inferred_task_failed_due_physical = True
    elif task_success is None:
        # Backward-compatible heuristic for callers that do not pass task outcome.
        if call_count > 0 and physical_failure_count == call_count:
            inferred_task_failed_due_physical = True

    # Calculate average call counts per role-step.
    # Use router decision-stage call_count so failures/retries do not inflate values.
    avg_role_call_counts = {}
    for role_name in ["planner", "coder", "debugger", "filter"]: # Assuming these are the roles
        total_steps_for_role = role_step_counts[role_name]
        total_decision_calls_for_role = role_decision_call_counts[role_name]
        avg_role_call_counts[f"avg_{role_name}_call_count"] = \
            total_decision_calls_for_role / total_steps_for_role if total_steps_for_role > 0 else 0


    total_latency = agent_total_latency + router_total_latency
    total_token_count = agent_total_tokens + router_token_count

    summary_record = {
        "session_id": str(session_id),
        "task_id": target_task_id,
        "step_id": "end",
        "agent_latency": agent_total_latency,
        "router_latency": router_total_latency,
        "network_latency": total_network_latency,
        "total_latency": total_latency,
        "agent_token_count": agent_total_tokens,
        "router_token_count": router_token_count,
        "total_token_count": total_token_count,
        "timestamp": time.time(),
        
        # New Metrics
        "call_count": call_count,
        "step_num": step_num,
        "network_failure_rate": network_failure_rate,
        "agent_failure_rate": agent_failure_rate, # New: Add agent failure rate
        "physical_failure_count": physical_failure_count,
        "physical_failure_attempt_rate": physical_failure_attempt_rate,
        "task_success": task_success,
        "task_fail_reason": task_fail_reason,
        "task_failed_due_physical": inferred_task_failed_due_physical,
        "planner_step_num": role_step_counts["planner"],
        "coder_step_num": role_agent_invocations["coder"], # Corrected to use agent invocations
        "debug_step_num": role_agent_invocations["debugger"], # Corrected to use agent invocations
        "filter_step_num": role_agent_invocations["filter"], # Corrected to use agent invocations
        **avg_role_call_counts, # Unpack average call counts for roles
    }

    with open(task_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
