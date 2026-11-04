from copy import deepcopy
import os
import yaml
from datawiseagent.common.config import global_config
from datawiseagent.common.log import logger
from datawiseagent.llms import load_llm, BaseChatModel


class AgentManager:
    """
    Manages the lifecycle and configuration of LLM clients.

    This class reads the global configuration, initializes all specified LLM clients,
    and provides access to them. It ensures that LLM clients are created only once.
    """

    def __init__(self):
        """Initializes the LLM manager by loading all models from config."""
        self._real_mode = bool(global_config.get("agent", {}).get("real_mode", False))
        self._network_mode = global_config.get("agent", {}).get(
            "network_mode", "real_probe" if self._real_mode else "simulated"
        )

        # 1. Load the general LLM configurations (the pool of all LLM definitions)
        self._llms_config = global_config.get(
            "llms", {"default_model": global_config.get("llm", {})}
        )

        # Load router name from global_config (named selection_strategy as per user)
        self._selection_strategy = global_config.get("agent", {}).get("selection_strategy", "bestone")
        logger.info(f"Router selection strategy set to: {self._selection_strategy}")

        # 2. Initialize all LLM clients based on _llms_config
        self._llms: dict[str, BaseChatModel] = {}
        for name, config in self._llms_config.items():
            logger.info(f"Initializing LLM client for config: {name}")
            self._llms[name] = load_llm(deepcopy(config))

        # 3. Determine role-based model configurations (from agent.yaml or fallback)
        self._agent_configs = {} # Renamed from _role_model_configs
        agent_yaml_loaded = False
        _agent_config_path = self._resolve_agent_config_path()
        try:
            logger.info(f"Attempting to load role-based agent configs from {_agent_config_path}")
            with open(_agent_config_path, 'r') as f:
                loaded_configs = yaml.safe_load(f)
                if loaded_configs and 'agent_configs' in loaded_configs:
                    self._agent_configs = loaded_configs['agent_configs']
                    self._normalize_agent_configs_for_mode(self._agent_configs)
                    agent_yaml_loaded = True
                    logger.info(f"Successfully loaded role-based agent configs from {_agent_config_path}.")
                else:
                    logger.warn(f"'{_agent_config_path}' found but 'agent_configs' section is missing or empty.")
        except FileNotFoundError:
            logger.warn(f"'{_agent_config_path}' not found. Falling back to global_config['agent']['llm_mapping'].")
        except Exception as e:
            logger.error(f"Error loading '{_agent_config_path}': {e}. Falling back to global_config['agent']['llm_mapping'].")

        if not agent_yaml_loaded:
            # Fallback: Construct _agent_configs from old llm_mapping
            _agent_global_config = global_config.get("agent", {})
            _llm_mapping = _agent_global_config.get("llm_mapping", {"default": "default_model"})
            logger.info(f"Constructing agent configs from old llm_mapping: {_llm_mapping}")
            for role, model_name in _llm_mapping.items():
                if model_name in self._llms_config:
                    # Construct a full LLM config for the agent from the global pool
                    # Assuming for fallback, the agent name is just the model_name
                    self._agent_configs[role] = {model_name: self._llms_config[model_name]}
                else:
                    logger.warn(f"Model name '{model_name}' from llm_mapping not found in main 'llms' config pool. Cannot create fallback config for role '{role}'.")

        # Determine the default LLM name for the AgentManager itself
        self._default_llm_name = "unknown_default" # Initialize with a safe default

        # Try to find a 'default' role in agent.yaml
        _default_agent_role_data = self._agent_configs.get('default', {})
        if _default_agent_role_data:
            first_model_name_in_default = next(iter(_default_agent_role_data))
            self._default_llm_name = first_model_name_in_default
        elif self._agent_configs: # If no 'default' role, but agent.yaml has other roles
            # Pick the first model from the first role defined in agent.yaml
            first_role_name = next(iter(self._agent_configs))
            first_role_models = self._agent_configs[first_role_name]
            if first_role_models:
                self._default_llm_name = next(iter(first_role_models))
        elif self._llms_config: # Fallback to the first LLM in the global pool
            self._default_llm_name = next(iter(self._llms_config.keys()))

        all_configured_agent_models = set()
        for role_configs in self._agent_configs.values():
            for model_name in role_configs.keys():
                all_configured_agent_models.add(model_name)
        
        logger.info(f"Default LLM set to: {self._default_llm_name}")
        config_source = (
            _agent_config_path
            if agent_yaml_loaded
            else "global_config['agent']['llm_mapping']"
        )
        logger.info(
            f"Successfully configured agent models from {config_source}: "
            f"{list(all_configured_agent_models)}"
        )
        logger.info(f"Global LLM pool (for fallback/other uses): {list(self._llms.keys())}")


    @property
    def llms_config(self) -> dict:
        """Returns the raw configuration dictionary for all LLMs."""
        return self._llms_config

    @property
    def agent_configs(self) -> dict: # New public property for agent configs
        """Returns the role-based agent configurations (from agent.yaml or fallback)."""
        return self._agent_configs
    
    @property
    def llms(self) -> dict[str, BaseChatModel]:
        """Returns the dictionary of initialized LLM clients."""
        return self._llms

    @property
    def default_llm(self) -> BaseChatModel:
        """Returns the default LLM client instance."""
        # Fallback to the first available client if the default name isn't in the pool
        return self._llms.get(self._default_llm_name, next(iter(self._llms.values()), None))

    @property
    def default_llm_config(self) -> dict:
        """Returns the configuration for the default LLM."""
        return self._llms_config.get(self._default_llm_name, {})

    @property
    def selection_strategy(self) -> str:
        """Returns the configured model selection strategy."""
        return self._selection_strategy

    @property
    def real_mode(self) -> bool:
        return self._real_mode

    @property
    def network_mode(self) -> str:
        return self._network_mode

    def _resolve_agent_config_path(self) -> str:
        """
        Resolve role-based agent config path.
        If `agent.agent_config_path` is not set, pick by `agent.real_mode`:
        - real_mode=True  -> configs/agent_real.yaml
        - real_mode=False -> configs/agent.yaml
        """
        agent_cfg = global_config.get("agent", {})
        configured_path = agent_cfg.get("agent_config_path")
        if not configured_path:
            configured_path = "configs/agent_real.yaml" if self._real_mode else "configs/agent.yaml"
        if os.path.isabs(configured_path):
            return configured_path
        _config_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(_config_dir)))
        return os.path.join(_project_root, configured_path)

    def _normalize_agent_configs_for_mode(self, agent_configs: dict) -> None:
        """
        Normalize per-agent configs in-place.
        In real mode:
        - Canonicalize Azure key names to base_url/key/version.
        - If Azure fields are present, switch llm_type to azure-openai-real-chat.
        """
        if not isinstance(agent_configs, dict):
            return
        for _, role_configs in agent_configs.items():
            if not isinstance(role_configs, dict):
                continue
            for _, config in role_configs.items():
                if not isinstance(config, dict):
                    continue
                if "endpoint" in config and "base_url" not in config:
                    config["base_url"] = config["endpoint"]
                if "subscription_key" in config and "key" not in config:
                    config["key"] = config["subscription_key"]
                if "api_version" in config and "version" not in config:
                    config["version"] = config["api_version"]

                if self._real_mode and config.get("base_url") and config.get("key"):
                    config["llm_type"] = "azure-openai-real-chat"


_agent_manager_instance = None

class LazyAgentManager:
    def __getattr__(self, name):
        global _agent_manager_instance
        if _agent_manager_instance is None:
            logger.info("First access to agent_pool_manager: Initializing AgentManager instance...")
            _agent_manager_instance = AgentManager()
        return getattr(_agent_manager_instance, name)

agent_pool_manager = LazyAgentManager()
