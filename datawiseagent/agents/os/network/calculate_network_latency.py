import json
import yaml
from typing import List, Dict, Any, Optional
import random # New: import random

# Assuming the NotebookCell structure is available from this path
from datawiseagent.common.types.cell import NotebookCell, MarkdownCell, CodeCell
from datawiseagent.common.config import global_config

_network_topology = None

def _get_network_topology() -> Dict[str, Any]:
    """
    Loads and caches the network topology from the global configuration.
    """
    global _network_topology
    if _network_topology is None:
        _network_topology = global_config.get("network_topology", {}).get("matrix", {})
        if not _network_topology:
            # 这是一个容错处理，防止配置为空时报错，返回空字典
            return {}
    return _network_topology

def _calculate_cells_size(cells: List[NotebookCell]) -> int:
    """
    Calculates the byte size of a list of NotebookCell objects by serializing to JSON.
    Returns: Size in Bytes.
    """
    if not cells:
        return 0

    try:
        # Convert Pydantic models to a list of dictionaries
        list_of_dicts = [cell.model_dump() for cell in cells]
        
        # Serialize to JSON string to estimate payload size
        json_string = json.dumps(list_of_dicts, default=str)
        
        # Calculate size in UTF-8 bytes
        byte_size = len(json_string.encode('utf-8'))
        
        return byte_size
    except Exception as e:
        raise TypeError(f"Failed to calculate size of NotebookCells: {e}")


def get_network_latency(
    source_zone: Optional[str],
    target_zone: Optional[str],
    cells: List[NotebookCell]
) -> tuple[float, int]: # Modified: return type hint
    """
    Calculates the network latency based on Bandwidth, Base Latency, Jitter, and Loss Rate.
    Simulates probabilistic packet loss and adds random jitter to latency.
    
    Formula: 
        Total Latency (s) = (Transmission Time (ms) + Propagation Delay (ms) + Jitter (ms)) / 1000.0
        Transmission Time (ms) = Data Size (Bytes) / (Bandwidth (tokens/ms) * 4 Bytes/Token)

    Args:
        source_zone: Origin zone (e.g., "Local"). If None, defaults to "Local".
        target_zone: Destination zone (e.g., "Global"). If None, defaults to "Local".
        cells: List of NotebookCell objects (the payload).

    Returns:
        tuple[float, int]: A tuple containing:
            - Total estimated latency in seconds, or -1.0 if packet loss occurs.
            - The calculated size of the cells in bytes.
    """
    # 0. Handle source_zone/target_zone being None and validate inputs
    if source_zone is None:
        source_zone = "Local"
    if target_zone is None:
        target_zone = "Local"
        
    if not isinstance(source_zone, str) or not isinstance(target_zone, str):
        raise TypeError("source_zone and target_zone must be strings after defaulting.")
    
    # Calculate the size of the data (Bytes) initially
    data_size_bytes = _calculate_cells_size(cells)

    # 1. Load network topology
    topology = _get_network_topology()
    if not topology:
        raise ValueError("Network topology configuration is missing or empty.")
    
    # 2. Look up connection details
    # Define canonical order to ensure "Local_to_Global" is same as "Global_to_Local"
    zone_order = ["Local", "Regional", "Global"]
    try:
        # Capitalize zones to ensure case-insensitive matching
        source_zone_cap = source_zone.capitalize()
        target_zone_cap = target_zone.capitalize()
        
        # Handle same zone connection, key format is just Zone_to_Zone
        if source_zone_cap == target_zone_cap:
            key = f"{source_zone_cap}_to_{source_zone_cap}"
        else:
            # For different zones, sort them to create a canonical key
            sorted_zones = sorted([source_zone_cap, target_zone_cap], key=lambda z: zone_order.index(z))
            key = f"{sorted_zones[0]}_to_{sorted_zones[1]}"
            
    except ValueError as e:
        raise ValueError(f"Invalid zone name. Must be one of {zone_order}. Got: {source_zone}, {target_zone}") from e
        
    connection_stats = topology.get(key)
    if connection_stats is None:
        raise ValueError(f"No network connection defined for '{source_zone}' to '{target_zone}' (key: {key})")

    # =========================================================================
    # CORE CALCULATION LOGIC (Probabilistic)
    # =========================================================================
    
    # Extract parameters
    base_latency_ms = connection_stats.get("base", 0)
    jitter_ms = connection_stats.get("jitter", 0)
    bandwidth_tokens_per_ms = connection_stats.get("bandwidth", 1) # Bandwidth in tokens/ms
    loss_rate = connection_stats.get("loss_rate", 0.0)

    # --- Step A: Simulate Packet Loss ---
    if random.random() < loss_rate:
        return -1.0, data_size_bytes # Packet lost

    # --- Step B: Calculate Transmission Time (发送时延) ---
    transmission_time_ms = 0.0
    if data_size_bytes > 0:
        if bandwidth_tokens_per_ms <= 0:
            return float('inf'), data_size_bytes # Infinite latency if bandwidth is zero

        # Assume 1 Token ≈ 4 Bytes
        # Transmission Time (ms) = Data Size (Bytes) / (Bandwidth (tokens/ms) * 4 Bytes/Token)
        transmission_time_ms = data_size_bytes / (bandwidth_tokens_per_ms * 4)

    # --- Step C: Calculate Propagation Delay with Jitter ---
    # Random jitter follows a normal distribution centered at 0 with sigma = jitter_ms.
    random_jitter_val = random.gauss(mu=0, sigma=jitter_ms)

    # Keep a small positive floor so negative jitter can reduce latency
    # without creating unrealistically fast near-zero links.
    minimum_latency_ms = max(1.0, 0.2 * base_latency_ms)
    current_latency_ms = max(minimum_latency_ms, base_latency_ms + random_jitter_val)

    # --- Step D: Total Latency (ms) ---
    total_latency_ms = transmission_time_ms + current_latency_ms
    
    return total_latency_ms / 1000.0, data_size_bytes # Convert to seconds


# =============================================================================
# Example Usage and Testing
# =============================================================================
if __name__ == "__main__":
    # Mocking the configuration for standalone testing
    # Using the parameters from your query: Base=800ms, Bandwidth=10Mbps
    if not global_config.get("network_topology"):
        print("Loading mock configuration for testing...")
        global_config["network_topology"] = {
            "zones": ["Global", "Regional", "Local"],
            "matrix": {
                # High Latency, Low Bandwidth (Your specific example)
                "Local_to_Global":    { "base": 800, "jitter": 150, "bandwidth": 10  },
                
                # Medium Latency, Medium Bandwidth
                "Local_to_Regional":  { "base": 150, "jitter": 30,  "bandwidth": 50  },
                
                # Low Latency, High Bandwidth
                "Local_to_Local":     { "base": 2,   "jitter": 1,   "bandwidth": 1000 } # 1Gbps
            }
        }

    print("\n--- Running Network Latency Calculation Tests ---")

    # Create a dummy file of approx 5MB to match your example calculation
    # 5MB = 5 * 1024 * 1024 bytes approx. 
    # Let's create a string content that approximates this size.
    target_size_mb = 5
    target_size_bytes = target_size_mb * 1024 * 1024
    
    # Creating a large mock content
    large_content = "x" * (target_size_bytes // 2) # Divide by 2 roughly as JSON overhead adds up
    
    mock_cells_large = [
        CodeCell(content=large_content, role="assistant", name="Datawise_Agent")
    ]
    
    # --- Test Case: Local to Global (The 800ms / 10Mbps link) ---
    source = "Local"
    target = "Global"
    
    try:
        real_size_bytes = _calculate_cells_size(mock_cells_large)
        real_size_mb = real_size_bytes / (1024 * 1024) # Size in Binary MB (MiB) for display
        
        # Note: In bandwidth calc, we usually use Decimal MB (10^6) or binary depending on definition.
        # The logic inside the function uses bits (size * 8) / (Mbps * 10^6), which is standard.
        
        print(f"Data Payload Size: {real_size_bytes} bytes ({real_size_mb:.2f} MiB)")
        print(f"Link Settings: Base=800ms, Bandwidth=10Mbps")
        
        latency = get_network_latency(source, target, mock_cells_large)
        
        print(f"Calculated Total Latency: {latency:.4f} seconds")
        
        # Verification check
        # Trans Delay = (Bytes * 8) / (10 * 1,000,000)
        expected_trans = (real_size_bytes * 8) / 10_000_000
        expected_net = 0.8 # 800ms
        print(f"  -> Expected Trans Delay: {expected_trans:.4f}s")
        print(f"  -> Expected Net Delay:   {expected_net:.4f}s")
        print(f"  -> Sum: {expected_trans + expected_net:.4f}s")
        
    except (ValueError, TypeError) as e:
        print(f"Error: {e}")

    print("-" * 20)
