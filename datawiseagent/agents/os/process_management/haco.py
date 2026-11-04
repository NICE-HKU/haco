import json
import traceback
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy
import math
import os
import statistics

from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager
from datawiseagent.common.log import logger
from datawiseagent.llms import load_llm
from datawiseagent.llms.utils.jsonrepair import JsonRepair, JsonRepairError
from datawiseagent.common.config import global_config
from datawiseagent.common.types.cell import CodeCell, CodeOutputCell, MarkdownCell, StepCell, UserCell
from datawiseagent.common.types.node import ExecutionNode, StepNode
from datawiseagent.agents.os.process_management.scoring_prompt import build_scoring_prompt
from datawiseagent.agents.os.experience.experience_collecting import _effective_agent_latency, _effective_network_latency
from datawiseagent.agents.os.network.real_network_latency import probe_latency_from_agent_config

                          
EXECUTION_STRATEGY = "wait_for_all"
                                                                        
TOKEN_TO_BYTES = 4.0
                                                                                  
DEFAULT_COLD_START_BW_EFFECTIVE = 650_000.0
LATENCY_DISCOUNT_ALPHA_DEFAULT = 0.5

                                                                        
                                                                                                                                                           
 
                 
         
 
                      
             
 
                
                                                                                                                                                                                                                                                      
 
                    
                                                                                                                                                                                                                                       
    
                                      
                             
                                                  
    
     

def _format_result_for_judge(result: Any) -> str:
    """
    Extract human-readable content from a StepNode/ExecutionNode result for the LLM judge.
    Avoids dumping raw Python repr (with UUIDs, duplicate CodeOutputCell, etc.).
    Falls back to str(result) if the result is not a recognized node type.
    """
    node = result
                                                          
    if isinstance(result, tuple) and len(result) == 2:
        node = result[1]

    if not isinstance(node, (ExecutionNode, StepNode)):
        return str(result)

    parts = []
    cells = node.cells_generated
    if not isinstance(cells, list):
        cells = [cells]

    seen_ids = set()
    for cell in cells:
        cell_id = getattr(cell, "id", None)
        if cell_id is not None:
            if cell_id in seen_ids:
                continue
            seen_ids.add(cell_id)

        if isinstance(cell, (StepCell, MarkdownCell)):
            parts.append(cell.content)
        elif isinstance(cell, CodeCell):
            parts.append(f"```python\n{cell.content}\n```")
            if cell.code_output is not None:
                output_content = cell.code_output.content
                                                                           
                if len(output_content) > 1000:
                    output_content = output_content[:1000] + "\n...(truncated)"
                parts.append(f"Code Output:\n{output_content}")
        elif isinstance(cell, CodeOutputCell):
                                                                                              
            pass

    return "\n\n".join(parts) if parts else str(result)


def _extract_action_signal(result: Any) -> Optional[str]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
        return result[0]
    if isinstance(result, (ExecutionNode, StepNode)):
        return result.action_signal
    return None


def _build_judge_context(base_judge_context: Optional[Dict[str, Any]], raw_result: Any) -> Dict[str, Any]:
    judge_context = dict(base_judge_context or {})
    current_action_signal = _extract_action_signal(raw_result)
    if current_action_signal:
        judge_context["current_action_signal"] = current_action_signal

    previous_winner_result = judge_context.pop("previous_winner_result", None)
    if previous_winner_result is not None:
        judge_context["previous_winner_response"] = _format_result_for_judge(previous_winner_result)
        previous_winner_action_signal = _extract_action_signal(previous_winner_result)
        if previous_winner_action_signal:
            judge_context["previous_winner_action_signal"] = previous_winner_action_signal

    return judge_context


                                                            

def _get_experience_dir() -> str:
    """Dynamically retrieves the experience directory from global_config."""
    return global_config.get("log", {}).get("experience_dir", "datawiseagent/agents/os/experience/")


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_haco_runtime_params() -> Dict[str, float]:
    """
    Load HACO runtime hyperparameters from agent.haco in config.
    Falls back to defaults when keys are missing or invalid.
    """
    agent_cfg = global_config.get("agent", {}) or {}
    haco_cfg = agent_cfg.get("haco", {}) or {}

    return {
        "exploration_factor": _safe_float(haco_cfg.get("exploration_factor"), 1.3),
        "risk_aversion_factor": _safe_float(haco_cfg.get("risk_aversion_factor"), 0.6),
        "jitter_penalty": _safe_float(haco_cfg.get("jitter_penalty"), 2.0),
        "reliability_target": _safe_float(haco_cfg.get("reliability_target"), 0.90),
        "latency_discount_alpha": _safe_float(
            haco_cfg.get("latency_discount_alpha"),
            LATENCY_DISCOUNT_ALPHA_DEFAULT,
        ),
    }

def _load_agent_profiles() -> Dict[str, Dict]:
    """Loads agent performance profiles from the experience log."""
    agent_log_path = os.path.join(_get_experience_dir(), "experience_agent.jsonl")
    profiles = {}
    if not os.path.exists(agent_log_path):
                                                                                            
        return profiles

    with open(agent_log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                agent_path = data.get("agent_path")
                if agent_path:
                    profiles[agent_path] = data
            except json.JSONDecodeError:
                continue
    return profiles

def _load_network_profiles() -> Dict:
    """Loads network link profiles from the experience log."""
    network_log_path = os.path.join(_get_experience_dir(), "experience_network.jsonl")
    profiles = {}
    if not os.path.exists(network_log_path):
                                                                                                
        return profiles
    try:
        with open(network_log_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"[HACO-Router] Failed to load network experience: {e}")
    return profiles

def _calculate_haco_selection(
    candidate_agents: List[Dict],
    source_agent_name: str,
    role: str,
    exploration_factor: float = 1.3,                            
    risk_aversion_factor: float = 0.6,                                    
    jitter_penalty: float = 2.0,
    reliability_target: float = 0.70,                           
    latency_discount_alpha: float = LATENCY_DISCOUNT_ALPHA_DEFAULT,
) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    """
    HACO (Heterogeneous Agent Coordination) Selection Algorithm:

    Uses Beta-UCB/LCB Logic:
    1. Phase 1 (Ranking): Sort by optimistic utility with a bounded log discount on latency.
    2. Phase 2 (Selection): Ensure Joint Reliability > 0.95 using (Mean - 1.0*Std).
    """
    logger.debug(f"--- HACO SELECTION [START] | Role: '{role}' | Target: {reliability_target} ---")

    agent_profiles = _load_agent_profiles()
    network_profiles = _load_network_profiles()
    agent_cfg = global_config.get("agent", {})
    real_mode = bool(agent_cfg.get("real_mode", False))
    network_mode = agent_cfg.get("network_mode", "real_probe" if real_mode else "simulated")
    use_real_probe = real_mode and network_mode == "real_probe"
    probe_cfg = agent_cfg.get("real_probe", {}) or {}

    ranked_agents = []
    cac_params_dict = {}
    epsilon = 1e-6
    candidate_metrics = []

                                                            
    for agent in candidate_agents:
        agent_name_raw = agent["name"]
        agent_name_full = f"{role}-{agent_name_raw}"
        profile = agent_profiles.get(agent_name_full, {})

                   
        avg_agent_latency = profile.get("avg_latency", 0)
        avg_token_count = profile.get("avg_token_count", 0)

                                             
                                        
        mu_obs = profile.get("avg_score", 0.6)                      
        n_i = profile.get("call_count", 0)

                                 
        alpha_val = (mu_obs * n_i) + 1.0
        beta_val = ((1.0 - mu_obs) * n_i) + 1.0

                                       
                                       
        beta_sum = alpha_val + beta_val
        beta_mean = alpha_val / beta_sum

                                                                       
        beta_var = (alpha_val * beta_val) / ((beta_sum ** 2) * (beta_sum + 1))
        beta_std = math.sqrt(beta_var)

                                             
                                                     
        agent_fail_rate = profile.get("agent_failure_rate", 0.0)

                 
        u, v = sorted([source_agent_name, agent_name_full])
        network_stats = network_profiles.get(u, {}).get(v, {})
        avg_bw = network_stats.get("avg_bandwidth", 0)
        bw_jitter = network_stats.get("bandwidth_jitter", 0)
        if use_real_probe:
            loss_rate = max(0.0, min(1.0, network_stats.get("probe_failure_rate", 0.0)))
        else:
            loss_rate = max(0.0, min(1.0, network_stats.get("loss_rate", 0.0)))

                                                         
                                   
        success_prob = (1.0 - agent_fail_rate) * (1.0 - loss_rate)

                 
        if use_real_probe:
            avg_rtt = float(network_stats.get("avg_rtt", 0) or 0)
            if avg_rtt <= 0:
                avg_rtt, probe_ok = probe_latency_from_agent_config(
                    agent.get("config"),
                    probe_samples=int(probe_cfg.get("samples", 3)),
                    probe_timeout_sec=float(probe_cfg.get("timeout_sec", 5.0)),
                    cache_ttl_sec=float(probe_cfg.get("cache_ttl_sec", 30.0)),
                    fallback_latency_sec=float(probe_cfg.get("fallback_latency_sec", 0.8)),
                )
                if not probe_ok:
                    loss_rate = max(loss_rate, 1.0)
            network_time_estimated = max(epsilon, float(avg_rtt))
        else:
            bw_effective = (
                max(epsilon, avg_bw - jitter_penalty * bw_jitter)
                if avg_bw > 0
                else DEFAULT_COLD_START_BW_EFFECTIVE
            )
            estimated_payload_bytes = (avg_token_count or 100) * TOKEN_TO_BYTES
            network_time_estimated = estimated_payload_bytes / bw_effective
        time_estimated = avg_agent_latency + network_time_estimated

                                            
                                       
        score_ucb = beta_mean + (exploration_factor * beta_std)
        score_ucb = min(1.0, score_ucb)             
        candidate_metrics.append(
            {
                "agent": agent,
                "name": agent_name_full,
                "beta_mean": beta_mean,
                "beta_std": beta_std,
                "p_success": success_prob,
                "score_ucb": score_ucb,
                "time_estimated": time_estimated,
                "agent_fail_rate": agent_fail_rate,
                "loss_rate": loss_rate,
                "alpha_val": alpha_val,
                "beta_val": beta_val,
            }
        )

    if not candidate_metrics:
        logger.debug("[HACO-Selection] No agents available.")
        return [], {}

    time_reference = statistics.median(
        max(epsilon, item["time_estimated"]) for item in candidate_metrics
    )

    for item in candidate_metrics:
        time_estimated = item["time_estimated"]
        score_ucb = item["score_ucb"]
        success_prob = item["p_success"]
        time_cost = math.log1p(time_estimated / max(epsilon, time_reference))
        latency_discount = 1.0 + (latency_discount_alpha * time_cost)

                                                                   
        rank_score = (score_ucb * success_prob) / latency_discount

        cac_params_dict[item["name"]] = {
            "Mean": item["beta_mean"],
            "Std": item["beta_std"],
            "Time": time_estimated,
            "TimeReference": time_reference,
            "TimeCost": time_cost,
            "LatencyDiscount": latency_discount,
            "P_Success": success_prob,
            "RankScore": rank_score,
            "AgentFailRate": item["agent_fail_rate"],
            "NetworkLossRate": item["loss_rate"],
            "ScoreUCB": score_ucb,
        }

        logger.debug(
            f"[HACO-Rank] Agent: {item['name']} | "
            f"Score: {rank_score:.4f} (UCB={score_ucb:.3f}, P_Succ={success_prob:.3f}, "
            f"T={time_estimated:.3f}, T0={time_reference:.3f}, D={latency_discount:.3f}) | "
            f"Beta(a={item['alpha_val']:.1f}, b={item['beta_val']:.1f})"
        )

        ranked_agents.append(
            {
                "agent": item["agent"],
                "rank_score": rank_score,
                "beta_mean": item["beta_mean"],
                "beta_std": item["beta_std"],
                "p_success": success_prob,
                "name": item["name"],
            }
        )
            
    ranked_agents.sort(key=lambda x: x["rank_score"], reverse=True)

                                                     
    selected_agents = []
    joint_failure_prob = 1.0

    for item in ranked_agents:
        selected_agents.append(item["agent"])

                                   
                                                        
        p_safe_quality = item["beta_mean"] - (risk_aversion_factor * item["beta_std"])
        p_safe_quality = max(0.1, p_safe_quality)              

                    
                                            
        node_reliability = p_safe_quality * item["p_success"]

                                                       
        node_reliability = min(node_reliability, 0.98)

                     
        joint_failure_prob *= (1 - node_reliability)

        current_reliability = 1 - joint_failure_prob
        logger.debug(
            f"[HACO-Select] Added '{item['name']}' | "
            f"Rel_Node: {node_reliability:.3f} (LCB={p_safe_quality:.3f}) | "
            f"Rel_Sys: {current_reliability:.5f} (Target: {reliability_target})"
        )

                     
        if current_reliability >= reliability_target:
            logger.info(f"[HACO-Selection] Target {reliability_target} met with {len(selected_agents)} agents.")
            break

    return selected_agents, cac_params_dict

def get_selected_agent_list(role: str) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    """Retrieves selected agent configurations using the HACO algorithm."""
    source_agent_name = "router"

    role_configs_dict = agent_pool_manager.agent_configs.get(role)
    if not role_configs_dict:
        role_configs_dict = agent_pool_manager.agent_configs.get('default', {})
        if not role_configs_dict:
            logger.warn(f"No agent configurations found for role '{role}'.")
            return [], {}

    candidate_agents = [{"name": name, "config": cfg} for name, cfg in role_configs_dict.items()]
    haco_params = _get_haco_runtime_params()

                                                             
    selected_agents, cac_params = _calculate_haco_selection(
        candidate_agents=candidate_agents,
        source_agent_name=source_agent_name,
        role=role,
        exploration_factor=haco_params["exploration_factor"],
        risk_aversion_factor=haco_params["risk_aversion_factor"],
        jitter_penalty=haco_params["jitter_penalty"],
        reliability_target=haco_params["reliability_target"],
        latency_discount_alpha=haco_params["latency_discount_alpha"],
    )

    logger.info(
        "[HACO-Router] Params | "
        f"reliability_target={haco_params['reliability_target']:.3f}, "
        f"risk_aversion_factor={haco_params['risk_aversion_factor']:.3f}, "
        f"exploration_factor={haco_params['exploration_factor']:.3f}, "
        f"jitter_penalty={haco_params['jitter_penalty']:.3f}, "
        f"latency_discount_alpha={haco_params['latency_discount_alpha']:.3f}"
    )
    logger.info(f"[HACO-Router] Selected {len(selected_agents)} agents: {[a['name'] for a in selected_agents]}")
    return selected_agents, cac_params

async def process_results(
    results: List[Any],
    role: str,
    query: str,
    step_info: str = "",
    judge_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[str], Optional[List[Dict]], int]:
    """Processes results from multiple agents and selects the best by ScoreUCB."""
    router_token_count = 0
    valid_results = [res for res in results if not isinstance(res, Exception) and res is not None]

    logger.debug(f"--- PROCESS_RESULTS | Role: '{role}' | Valid: {len(valid_results)}/{len(results)} ---")

    if not valid_results:
        error_res = next((res for res in results if isinstance(res, Exception)), None)
        logger.warning(
            f"[ROUTER_AGENT_SCORES_NULL] router=haco reason=no_valid_results "
            f"role={role} total_results={len(results)}"
        )
        return error_res, None, None, router_token_count

    try:
        aggregator_config = agent_pool_manager.agent_configs['router']['router']
        aggregator_llm = load_llm(deepcopy(aggregator_config))

                                                                               
        scores_list = []
        for agent_run_details in valid_results:
            agent_name = agent_run_details["name"]
            raw_result = agent_run_details['result']
            formatted_result = _format_result_for_judge(raw_result)
            single_response = f"Agent '{agent_name}':\n{formatted_result}\n---\n"
            prompt = build_scoring_prompt(
                role=role,
                query=query,
                responses=single_response,
                step_info=step_info,
                judge_context=_build_judge_context(judge_context, raw_result),
            )

            log_header = f"router=haco agent={agent_name} role={role}"
            if step_info:
                log_header += f" | {step_info}"
            _judge_log_path = os.path.join(_get_experience_dir(), f"judge_debug_{role}.log")
            os.makedirs(os.path.dirname(_judge_log_path), exist_ok=True)
            with open(_judge_log_path, "a", encoding="utf-8") as _f:
                _f.write(f"\n===== [JUDGE INPUT] {log_header} =====\n{prompt}\n")

            llm_result = await aggregator_llm.agenerate_response(append_prompt=prompt)
            router_token_count += llm_result.total_tokens

            with open(_judge_log_path, "a", encoding="utf-8") as _f:
                _f.write(f"===== [JUDGE OUTPUT] {log_header} =====\n{llm_result.content}\n")

            try:
                repaired_json = JsonRepair(llm_result.content).repair()
                parsed_scores = json.loads(repaired_json)
                if not isinstance(parsed_scores, list):
                    parsed_scores = parsed_scores.get('results', []) or parsed_scores.get('evaluations', [])

                parsed_item = parsed_scores[0] if parsed_scores else {}
                parsed_score = parsed_item.get("score", 0)
                if not isinstance(parsed_score, (int, float)):
                    parsed_score = 0

                scores_list.append(
                    {
                        "agent_name": agent_name,
                        "score": parsed_score,
                        "reasoning": parsed_item.get("reasoning", ""),
                    }
                )
            except Exception as parse_err:
                logger.error("Failed to parse evaluation scores.")
                logger.warning(
                    f"[ROUTER_AGENT_SCORES_NULL] router=haco reason=score_parse_failed "
                    f"role={role} valid_results={len(valid_results)} target_agent={agent_name} "
                    f"parse_error={type(parse_err).__name__}: {parse_err} "
                    f"raw_output={llm_result.content[:500]!r}"
                )
                scores_list.append(
                    {
                        "agent_name": agent_name,
                        "score": None,
                        "reasoning": "parse_failed",
                    }
                )

                                                     
                                             
        best_agent_name = ""
        best_score_ucb = float('-inf')
        min_effective_cost = float('inf')
        results_map = {res["name"]: res for res in valid_results}

        for agent_run_details in valid_results:
            agent_name = agent_run_details["name"]
            agent_cac_data = agent_run_details.get("cac_data")
            score_ucb = float('-inf')
            if isinstance(agent_cac_data, dict):
                score_ucb = agent_cac_data.get("ScoreUCB", float('-inf'))

            total_effective_cost = 0.0
            for attempt in agent_run_details.get("attempts", []):
                attempt_type = attempt.get("type")
                effective_a_lat = _effective_agent_latency(
                    attempt.get("latency", 0.0),
                    attempt_type if attempt_type == 'agent_failure_rate' else None
                )
                effective_n_lat = _effective_network_latency(
                    attempt.get("network_latency", 0.0)
                )
                total_effective_cost += effective_a_lat + effective_n_lat

            if (
                score_ucb > best_score_ucb or
                (score_ucb == best_score_ucb and total_effective_cost < min_effective_cost)
            ):
                best_score_ucb = score_ucb
                min_effective_cost = total_effective_cost
                best_agent_name = agent_name

        if not best_agent_name:
            fallback_result = valid_results[0]
            return fallback_result["result"], fallback_result["name"], scores_list, router_token_count

        best_result = results_map[best_agent_name]["result"]
        logger.info(
            f"[HACO-Router] Final winner: {best_agent_name} "
            f"(ScoreUCB: {best_score_ucb:.4f}, Effective Cost: {min_effective_cost:.4f})"
        )

        return best_result, best_agent_name, scores_list, router_token_count

    except Exception as e:
        logger.error(f"Router processing failed: {e}\n{traceback.format_exc()}")
        logger.warning(
            f"[ROUTER_AGENT_SCORES_NULL] router=haco reason=router_processing_failed "
            f"role={role} valid_results={len(valid_results)} error={type(e).__name__}: {e}"
        )
        fallback_result = valid_results[0]
        return fallback_result["result"], fallback_result["name"], None, router_token_count
