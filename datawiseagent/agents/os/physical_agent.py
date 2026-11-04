from __future__ import annotations
import importlib
from copy import deepcopy
from datawiseagent.llms import BaseChatModel, load_llm
from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager
from datawiseagent.common.log import logger

class PhysicalAgent:
    """一个包含唯一名称和特定角色LLM客户端的包装器。"""
    def __init__(self, name: str, llm_client: BaseChatModel):
        self.name = name
        self.llm = llm_client

    @classmethod
    def get_physical_agent(cls, role: str) -> PhysicalAgent:
        """
        工厂方法，动态加载路由器以选择一个具名代理配置，
        然后创建一个带有唯一名称的 PhysicalAgent 实例。
        """
        router_name = agent_pool_manager.selection_strategy
        try:
            module_path = f"datawiseagent.agents.os.process_management.{router_name}"
            router_module = importlib.import_module(module_path)
            
            selected_key, full_llm_config_dict = router_module.get_selected_agent_list(role)
            
            if not selected_key or not full_llm_config_dict:
                raise ValueError(f"Router failed to select an agent for role '{role}'.")

            # Construct the agent's unique name
            agent_name = f"{role}-{selected_key}"

            # Directly use the full_llm_config_dict for loading the LLM
            llm_client = load_llm(deepcopy(full_llm_config_dict))
            
            return cls(name=agent_name, llm_client=llm_client)
        
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent creation for role '{role}': {e}")
            logger.warn("Falling back to agent pool's default LLM with a default name.")
            return cls(name=f"{role}-default-fallback", llm_client=agent_pool_manager.default_llm)
