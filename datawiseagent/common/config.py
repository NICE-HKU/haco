import yaml
import os
from .log import logger


def _update_config(default, user):
    for key, value in user.items():
        if isinstance(value, dict):
            default[key] = _update_config(default.get(key, {}), value)
        else:
            default[key] = value
    return default


def update_custom_config(config):
    if custom_config_path := os.getenv("CUSTOM_CONFIG", ""):
        logger.info("Loading custom config from %s" % custom_config_path)
        custom_config: dict = yaml.safe_load(open(custom_config_path))

        _update_config(config, custom_config)


def _resolve_agent_config_path() -> str:
    """
    Resolve role-based agent config path.

    Priority:
    1) `agent.agent_config_path` from merged global config
    2) `agent.real_mode` default path switch:
       - True  -> configs/agent_real.yaml
       - False -> configs/agent.yaml
    """
    agent_cfg = global_config.get("agent", {}) if isinstance(global_config, dict) else {}
    configured_path = agent_cfg.get("agent_config_path")
    if not configured_path:
        configured_path = "configs/agent_real.yaml" if agent_cfg.get("real_mode") else "configs/agent.yaml"
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(_project_root, configured_path)


# Construct an absolute path to the default config file to avoid CWD issues
_config_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_config_dir))
_default_config_path = os.path.join(_project_root, "configs", "default_config.yaml")

try:
    logger.info(f"Loading default config from {_default_config_path}")
    global_config = yaml.safe_load(open(_default_config_path))
except FileNotFoundError:
    logger.warn(
        "No default config file found, using empty config as the default config. If you want to use a default config file, please make sure it is in the 'configs' folder and named 'default_config.yaml'"
    )
    global_config = {}
update_custom_config(global_config)

# Load network_topology from agent config and merge it into global_config.
# Real mode can point to agent_real.yaml; when that file does not define
# network_topology, fall back to the canonical configs/agent.yaml.
_candidate_paths = [_resolve_agent_config_path()]
_fallback_agent_yaml = os.path.join(_project_root, "configs", "agent.yaml")
if _fallback_agent_yaml not in _candidate_paths:
    _candidate_paths.append(_fallback_agent_yaml)

for _agent_config_path in _candidate_paths:
    try:
        logger.info(f"Loading network_topology from agent config at {_agent_config_path}")
        agent_full_config = yaml.safe_load(open(_agent_config_path))
        if agent_full_config and "network_topology" in agent_full_config:
            global_config.setdefault("network_topology", {})
            _update_config(global_config["network_topology"], agent_full_config["network_topology"])
            break
    except FileNotFoundError:
        logger.warn(f"Agent config not found at {_agent_config_path}, skipping network_topology load from this file.")
    except Exception as e:
        logger.error(f"Error loading or merging network_topology from {_agent_config_path}: {e}")
