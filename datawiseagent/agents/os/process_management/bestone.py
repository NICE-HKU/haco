import json
import os
import random
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional

from datawiseagent.common.types.cell import CodeCell, CodeOutputCell, MarkdownCell, StepCell, UserCell
from datawiseagent.common.types.node import ExecutionNode, StepNode

from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager
from datawiseagent.common.config import global_config
from datawiseagent.common.log import logger
from datawiseagent.llms import load_llm
from datawiseagent.llms.utils.jsonrepair import JsonRepair
from datawiseagent.agents.os.process_management.scoring_prompt import build_scoring_prompt


EXECUTION_STRATEGY = "wait_for_all"

                                                                        
                                                                                                                                                           
 
                 
         
 
                      
             
 
                
                                                                                                                                                                                                                                                      
 
                    
                                                                                                                                                                                                                                       
    
                                      
                             
                                                  
    
     


def _get_experience_dir() -> str:
    """Dynamically retrieves the experience directory from global_config."""
    return global_config.get("log", {}).get("experience_dir", "datawiseagent/agents/os/experience/")


def _load_agent_profiles() -> Dict[str, Dict]:
    """Loads per-agent aggregated profiles from experience_agent.jsonl."""
    agent_log_path = os.path.join(_get_experience_dir(), "experience_agent.jsonl")
    profiles: Dict[str, Dict] = {}

    if not os.path.exists(agent_log_path):
        return profiles

    try:
        with open(agent_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                agent_path = record.get("agent_path")
                if agent_path:
                    profiles[agent_path] = record
    except Exception as e:
        logger.error(f"[BestOne-Router] Failed to load agent profiles: {e}")

    return profiles


def _get_baseline_exploration_params() -> Tuple[int, float]:
    """
    Returns (min_call_count_per_agent, epsilon) for bestone baseline.
    Values can be overridden from config:
      agent.bestone_min_call_count
      agent.bestone_epsilon
    """
    agent_cfg = global_config.get("agent", {})
    min_call_count = int(agent_cfg.get("bestone_min_call_count", 2))
    epsilon = float(agent_cfg.get("bestone_epsilon", 0.1))
    epsilon = min(1.0, max(0.0, epsilon))
    min_call_count = max(0, min_call_count)
    return min_call_count, epsilon


def _format_result_for_judge(result: Any) -> str:
    """
    Extract human-readable content from a StepNode/ExecutionNode result for the LLM judge.
    Avoids dumping raw Python repr (with UUIDs, duplicate CodeOutputCell, etc.).
    Falls back to str(result) if the result is not a recognized node type.
    """
    node = result
                                                          
    if isinstance(result, tuple) and len(result) == 2:
        node = result[1]

    if not isinstance(node, ExecutionNode):
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


def get_selected_agent_list(role: str) -> List[Dict]:
    """
    Select exactly one agent by the highest completion score (avg_score).
    This router ignores network metrics and failure-rate metrics by design.
    """
    role_configs_dict = agent_pool_manager.agent_configs.get(role)
    if not role_configs_dict:
        role_configs_dict = agent_pool_manager.agent_configs.get("default", {})
        if not role_configs_dict:
            logger.warning(f"[BestOne-Router] No agent configs found for role '{role}' or 'default'.")
            return []

    profiles = _load_agent_profiles()
    candidates = []
    for agent_name, config in role_configs_dict.items():
        full_name = f"{role}-{agent_name}"
        profile = profiles.get(full_name, {})
                                                                
        avg_score = float(profile.get("avg_score", 1.0))
        candidates.append(
            {
                "name": agent_name,
                "config": config,
                "avg_score": avg_score,
                "call_count": int(profile.get("call_count", 0)),
            }
        )

    if not candidates:
        return []

    min_call_count, epsilon = _get_baseline_exploration_params()

                                                                                              
    cold_candidates = [c for c in candidates if c["call_count"] < min_call_count]
    if cold_candidates:
        selected = random.choice(cold_candidates)
        logger.info(
            f"[BestOne-Router] Cold-start pick for role '{role}': '{selected['name']}' "
            f"(call_count={selected['call_count']} < min_call_count={min_call_count})."
        )
        return [{"name": selected["name"], "config": selected["config"]}]

                                                                          
    max_score = max(c["avg_score"] for c in candidates)
    top_candidates = [c for c in candidates if c["avg_score"] == max_score]
    best_candidate = random.choice(top_candidates)

    selected = best_candidate
    if len(candidates) > 1 and random.random() < epsilon:
        non_best_candidates = [c for c in candidates if c["name"] != best_candidate["name"]]
        if non_best_candidates:
            selected = random.choice(non_best_candidates)
            logger.info(
                f"[BestOne-Router] Explore pick for role '{role}': '{selected['name']}' "
                f"(epsilon={epsilon:.3f})."
            )
        else:
            logger.info(
                f"[BestOne-Router] Explore triggered but only one distinct candidate for role '{role}'."
            )
    else:
        logger.info(
            f"[BestOne-Router] Exploit pick for role '{role}': '{selected['name']}' "
            f"(avg_score={selected['avg_score']:.4f})."
        )

    return [{"name": selected["name"], "config": selected["config"]}]


async def process_results(
    results: List[Any],
    role: str,
    query: str = "",
    step_info: str = "",
    judge_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[str], Optional[List[Dict]], int]:
    """
    Process router results and forward one final result.
    Interface aligned with concurrent router contract:
    (final_result, selected_agent_name, agent_score_list, router_token_count)
    """
    router_token_count = 0
    valid_results = [res for res in results if not isinstance(res, Exception) and res is not None]

    if not valid_results:
        error_res = next((res for res in results if isinstance(res, Exception)), None)
        logger.warning(f"[BestOne-Router] No valid results for role '{role}'.")
        return error_res, None, None, router_token_count

    try:
                                                         
                                                                        
        results_map = {r["name"]: r["result"] for r in valid_results}

        aggregator_config = agent_pool_manager.agent_configs["router"]["router"]
        aggregator_llm = load_llm(deepcopy(aggregator_config))

                                                                                    
        scores_list = []
        for agent_run_details in valid_results:
            agent_name = agent_run_details["name"]
            raw_result = agent_run_details["result"]
            formatted_result = _format_result_for_judge(raw_result)
            single_response = f"Agent '{agent_name}':\n{formatted_result}\n---\n"
            prompt = build_scoring_prompt(
                role=role,
                query=query,
                responses=single_response,
                step_info=step_info,
                judge_context=_build_judge_context(judge_context, raw_result),
            )

            log_header = f"router=bestone agent={agent_name} role={role}"
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
                repaired = JsonRepair(llm_result.content).repair()
                parsed_scores = json.loads(repaired)
                if not isinstance(parsed_scores, list):
                    parsed_scores = parsed_scores.get("results", []) or parsed_scores.get("evaluations", [])
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
                    f"[ROUTER_AGENT_SCORES_NULL] router=bestone reason=score_parse_failed "
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
        best_score = float("-inf")
        for item in scores_list:
            score = item.get("score")
            a_name = item.get("agent_name", "")
            if a_name in results_map and isinstance(score, (int, float)) and score > best_score:
                best_score = float(score)
                best_agent_name = a_name

        if best_agent_name:
            logger.info(
                f"[BestOne-Router] LLM selected '{best_agent_name}' for role '{role}' "
                f"with best_score={best_score:.2f}."
            )
            return results_map[best_agent_name], best_agent_name, scores_list, router_token_count

                                                                                       
        fallback_name = valid_results[0]["name"]
        fallback_val = valid_results[0]["result"]
        logger.warning(
            f"[BestOne-Router] Falling back to first valid result for role '{role}' "
            f"from agent '{fallback_name}'."
        )
        return fallback_val, fallback_name, scores_list, router_token_count

    except Exception as e:
        fallback_name = valid_results[0]["name"]
        fallback_val = valid_results[0]["result"]
        logger.error(
            f"[BestOne-Router] Router processing failed for role '{role}': {e}. "
            f"Falling back to '{fallback_name}'."
        )
        return fallback_val, fallback_name, None, router_token_count
