import json
import os
import traceback
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional

from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager
from datawiseagent.common.config import global_config
from datawiseagent.common.log import logger
from datawiseagent.llms import load_llm
from datawiseagent.llms.utils.jsonrepair import JsonRepair, JsonRepairError
from datawiseagent.common.types.cell import CodeCell, CodeOutputCell, MarkdownCell, StepCell, UserCell
from datawiseagent.common.types.node import ExecutionNode, StepNode
from datawiseagent.agents.os.process_management.scoring_prompt import build_scoring_prompt


EXECUTION_STRATEGY = "wait_for_all"

                                                                        
                                                                                                                                                           
 
                 
         
 
                      
             
 
                
                                                                                                                                                                                                                                                      
 
                    
                                                                                                                                                                                                                                       
    
                                      
                             
                                                  
    
 
          
   
      
                                  
                 
                                                                                                                 
       
      
                                     
                 
                                                                                                                 
      
   
     


def _get_experience_dir() -> str:
    """Dynamically retrieves the experience directory from global_config."""
    return global_config.get("log", {}).get("experience_dir", "datawiseagent/agents/os/experience/")


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


def _build_judge_context(
    base_judge_context: Optional[Dict[str, Any]],
    raw_result: Any,
) -> Dict[str, Any]:
    judge_context = dict(base_judge_context or {})
    current_action_signal = _extract_action_signal(raw_result)
    if current_action_signal:
        judge_context["current_action_signal"] = current_action_signal

    previous_winner_result = judge_context.pop("previous_winner_result", None)
    if previous_winner_result is not None:
        judge_context["previous_winner_response"] = _format_result_for_judge(
            previous_winner_result
        )
        previous_winner_action_signal = _extract_action_signal(previous_winner_result)
        if previous_winner_action_signal:
            judge_context["previous_winner_action_signal"] = (
                previous_winner_action_signal
            )

    return judge_context


def get_selected_agent_list(role: str) -> List[Dict]:
    """
    Retrieves all agent configurations for a given role without any selection logic.

    Args:
        role: The role to get agents for (e.g., 'coder', 'planner').

    Returns:
        A list of dictionaries, where each dictionary contains
        the 'name' and 'config' of an agent.
    """
    role_configs_dict = agent_pool_manager.agent_configs.get(role)
    if not role_configs_dict:
        role_configs_dict = agent_pool_manager.agent_configs.get('default', {})
        if not role_configs_dict:
            logger.warning(f"No agent configurations found for role '{role}' or default role.")
            return []

    all_agents = []
    for agent_name, config in role_configs_dict.items():
        all_agents.append({
            "name": agent_name,
            "config": config
        })

    logger.info(f"Router found {len(all_agents)} agents for role '{role}'.")
    return all_agents


async def process_results(
    results: List[Any],
    role: str,
    query: str = "",
    step_info: str = "",
    judge_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[str], Optional[List[Dict]], int]:
    """
    Processes results from multiple agents and selects the best one via LLM scoring.
    Interface aligned with concurrent router contract:
    (final_result, selected_agent_name, agent_score_list, router_token_count)

    Args:
        results: List of dicts {"name": ..., "result": ..., "attempts": ..., "cac_data": ...}
                 or Exception objects.
        role: The role for which the agents were run.
        query: The original query / context for the judge.
        step_info: Optional context describing which step this is.

    Returns:
        (final_result, best_agent_name, scores_list, router_token_count)
    """
    router_token_count = 0

    valid_results = [res for res in results if not isinstance(res, Exception) and res is not None]

    if not valid_results:
        error_res = next((res for res in results if isinstance(res, Exception)), None)
        logger.warning(f"[MoA-Router] No valid results for role '{role}'.")
        return error_res, None, None, router_token_count

    logger.info(f"[MoA-Router] {len(valid_results)} valid result(s) for role '{role}'. Using LLM to score and select the best one.")

    try:
        results_map = {r["name"]: r["result"] for r in valid_results}

        aggregator_config = agent_pool_manager.agent_configs['router']['router']
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

            log_header = f"router=moa agent={agent_name} role={role}"
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
                    f"[ROUTER_AGENT_SCORES_NULL] router=moa reason=score_parse_failed "
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
                f"[MoA-Router] LLM selected '{best_agent_name}' for role '{role}' "
                f"with best_score={best_score:.2f}."
            )
            return results_map[best_agent_name], best_agent_name, scores_list, router_token_count

                                                                                       
        fallback_name = valid_results[0]["name"]
        fallback_val = valid_results[0]["result"]
        logger.warning(
            f"[MoA-Router] Falling back to first valid result for role '{role}' "
            f"from agent '{fallback_name}'."
        )
        return fallback_val, fallback_name, scores_list, router_token_count

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"[MoA-Router] Router processing failed for role '{role}': {e}\n{error_trace}")
        fallback_name = valid_results[0]["name"]
        fallback_val = valid_results[0]["result"]
        return fallback_val, fallback_name, None, router_token_count
