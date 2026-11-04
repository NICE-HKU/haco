import json
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from datawiseagent.common.config import global_config
from datawiseagent.common.types.cell import NotebookCell

_probe_cache: Dict[Tuple[str, str], Tuple[float, float, bool]] = {}


def _get_probe_cfg() -> Dict[str, float]:
    agent_cfg = global_config.get("agent", {})
    probe_cfg = agent_cfg.get("real_probe", {}) or {}
    return {
        "samples": int(probe_cfg.get("samples", 3)),
        "timeout_sec": float(probe_cfg.get("timeout_sec", 5.0)),
        "cache_ttl_sec": float(probe_cfg.get("cache_ttl_sec", 30.0)),
        "fallback_latency_sec": float(probe_cfg.get("fallback_latency_sec", 0.8)),
    }


def _calculate_cells_size(cells: List[NotebookCell]) -> int:
    if not cells:
        return 0
    list_of_dicts = [cell.model_dump() for cell in cells]
    json_string = json.dumps(list_of_dicts, default=str)
    return len(json_string.encode("utf-8"))


def _extract_azure_conn_fields(agent_config: Optional[Dict[str, Any]]) -> Tuple[str, str, str]:
    cfg = agent_config or {}
    base_url = cfg.get("base_url") or cfg.get("endpoint") or ""
    key = cfg.get("key") or cfg.get("subscription_key") or ""
    version = cfg.get("version") or cfg.get("api_version") or "2024-12-01-preview"
    return base_url, key, version


def probe_latency_from_agent_config(
    agent_config: Optional[Dict[str, Any]],
    *,
    probe_samples: Optional[int] = None,
    probe_timeout_sec: Optional[float] = None,
    cache_ttl_sec: Optional[float] = None,
    fallback_latency_sec: Optional[float] = None,
) -> Tuple[float, bool]:
    """
    Probe endpoint RTT using Azure deployment listing endpoint.
    Returns (latency_sec, probe_ok).
    """
    cfg = _get_probe_cfg()
    samples = probe_samples if probe_samples is not None else cfg["samples"]
    timeout_sec = probe_timeout_sec if probe_timeout_sec is not None else cfg["timeout_sec"]
    ttl_sec = cache_ttl_sec if cache_ttl_sec is not None else cfg["cache_ttl_sec"]
    fallback = fallback_latency_sec if fallback_latency_sec is not None else cfg["fallback_latency_sec"]

    base_url, key, version = _extract_azure_conn_fields(agent_config)
    if not base_url:
        return fallback, False

    cache_key = (base_url.rstrip("/"), str(version))
    now = time.time()
    cached = _probe_cache.get(cache_key)
    if cached and now - cached[0] <= ttl_sec:
        return cached[1], cached[2]

    url = f"{base_url.rstrip('/')}/openai/deployments?api-version={version}"
    headers = {"api-key": key} if key else {}
    samples_ms: List[float] = []

    for _ in range(max(1, int(samples))):
        start = time.perf_counter()
        try:
            # Any HTTP response means network RTT is measurable.
            requests.get(url, headers=headers, timeout=timeout_sec)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples_ms.append(elapsed_ms)
        except requests.RequestException:
            continue

    if not samples_ms:
        latency = max(0.001, float(fallback))
        _probe_cache[cache_key] = (now, latency, False)
        return latency, False

    latency = statistics.median(samples_ms) / 1000.0
    latency = max(0.001, float(latency))
    _probe_cache[cache_key] = (now, latency, True)
    return latency, True


def get_real_network_latency(
    source_zone: Optional[str],
    target_zone: Optional[str],
    cells: List[NotebookCell],
    *,
    target_agent_config: Optional[Dict[str, Any]],
    probe_samples: Optional[int] = None,
    probe_timeout_sec: Optional[float] = None,
    cache_ttl_sec: Optional[float] = None,
    fallback_latency_sec: Optional[float] = None,
) -> Tuple[float, int, bool]:
    """
    Real network latency path:
    - latency from RTT probe to target model endpoint
    - cell size measured for logging consistency
    """
    del source_zone, target_zone  # zone is unused in real probe mode
    cell_size = _calculate_cells_size(cells)
    latency, probe_ok = probe_latency_from_agent_config(
        target_agent_config,
        probe_samples=probe_samples,
        probe_timeout_sec=probe_timeout_sec,
        cache_ttl_sec=cache_ttl_sec,
        fallback_latency_sec=fallback_latency_sec,
    )
    return latency, cell_size, probe_ok
