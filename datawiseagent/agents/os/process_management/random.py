import random
from typing import List, Dict, Any, Tuple, Optional

from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager
from datawiseagent.common.log import logger


EXECUTION_STRATEGY = "wait_for_all"


def get_selected_agent_list(role: str) -> List[Dict]:
    """
    Randomly selects one agent for the given role.
    Returns a single-item list to fit concurrent executor expectations.
    """
    role_configs_dict = agent_pool_manager.agent_configs.get(role)
    if not role_configs_dict:
        role_configs_dict = agent_pool_manager.agent_configs.get("default", {})
        if not role_configs_dict:
            logger.warning(f"[Random-Router] No agent configs found for role '{role}' or 'default'.")
            return []

    selected_name = random.choice(list(role_configs_dict.keys()))
    selected_config = role_configs_dict[selected_name]
    model_name_log = selected_config.get("model", "Unknown")
    logger.info(
        f"[Random-Router] Role '{role}' selected agent '{selected_name}' "
        f"with model '{model_name_log}' randomly."
    )
    return [{"name": selected_name, "config": selected_config}]


async def process_results(
    results: List[Any],
    role: str,
    query: str = "",
) -> Tuple[Any, Optional[str], Optional[List[Dict]], int]:
    """
    Returns the first valid agent result.
    Interface is aligned with concurrent router contracts:
    (final_result, selected_agent_name, agent_score_list, router_token_count)
    """
    router_token_count = 0
    valid_results = [res for res in results if not isinstance(res, Exception) and res is not None]

    if not valid_results:
        error_res = next((res for res in results if isinstance(res, Exception)), None)
        logger.warning(f"[Random-Router] No valid results for role '{role}'.")
        return error_res, None, None, router_token_count

    if len(valid_results) > 1:
        logger.warning(
            f"[Random-Router] Expected one result for role '{role}', "
            f"but got {len(valid_results)}. Returning the first valid one."
        )

    selected_run = valid_results[0]
    if isinstance(selected_run, dict):
        selected_name = selected_run.get("name")
        selected_result = selected_run.get("result")
    elif isinstance(selected_run, (list, tuple)) and len(selected_run) >= 2:
                                                                  
        selected_name, selected_result = selected_run[0], selected_run[1]
    else:
        logger.warning(
            f"[Random-Router] Unsupported result payload for role '{role}': "
            f"type={type(selected_run).__name__}. Returning None."
        )
        return None, None, None, router_token_count

    logger.info(f"[Random-Router] Forwarding result from agent '{selected_name}'.")
    return selected_result, selected_name, None, router_token_count
