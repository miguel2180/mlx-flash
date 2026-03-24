import math
from typing import Dict, List, Any

def get_next_precision(b: int) -> int:
    """Returns the next valid quantization bit-width step."""
    if b <= 2: return 3
    if b == 3: return 4
    if b == 4: return 6
    if b >= 6: return 8
    return b

def get_bytes_per_param(bits: int) -> float:
    """Approximates the storage cost per parameter including block scales."""
    # Assuming block size 32. 
    # Q4_0: 16 bytes data + 2 byte scale = 18 bytes / 32 params = 0.5625 bytes/param
    # Q8_0: 32 bytes data + 2 byte scale = 34 bytes / 32 params = 1.0625 bytes/param
    block_size = 32
    scale_bytes = 2
    
    data_bytes = (bits * block_size) / 8.0
    total_bytes = data_bytes + scale_bytes
    return total_bytes / block_size

def allocate_bits(tensors: List[Dict[str, Any]], target_budget_bytes: int, min_bits: int = 3, max_bits: int = 8) -> Dict[str, int]:
    """
    Distributes bits across tensors based on their sensitivity score to maximize quality
    while strictly staying under target_budget_bytes.
    
    tensors: list of dicts [{'name': str, 'shape': tuple, 'sensitivity': float}]
    """
    current_budget = 0.0
    allocations = {}
    
    # 1. Start everything at minimum precision to establish baseline
    for t in tensors:
        num_params = math.prod(t['shape'])
        allocations[t['name']] = min_bits
        current_budget += num_params * get_bytes_per_param(min_bits)
        
    if current_budget > target_budget_bytes:
        raise ValueError(f"Target budget ({target_budget_bytes/1e9:.2f} GB) too strict even at {min_bits}-bit. Minimum required: {current_budget/1e9:.2f} GB")
        
    # 2. Sort tensors by sensitivity (descending)
    sorted_tensors = sorted(tensors, key=lambda x: x.get('sensitivity', 0.0), reverse=True)
    
    # 3. Greedily upgrade precision of the most sensitive layers
    budget_remaining = target_budget_bytes - current_budget
    
    while budget_remaining > 0:
        upgraded_any = False
        for t in sorted_tensors:
            name = t['name']
            current_bits = allocations[name]
            
            next_bits = get_next_precision(current_bits)
            
            if next_bits > max_bits or next_bits == current_bits:
                continue
                
            num_params = math.prod(t['shape'])
            
            # Calculate exact byte cost to upgrade this tensor to the next precision tier
            current_cost = num_params * get_bytes_per_param(current_bits)
            next_cost = num_params * get_bytes_per_param(next_bits)
            upgrade_cost_bytes = next_cost - current_cost
            
            if upgrade_cost_bytes <= budget_remaining:
                allocations[name] = next_bits
                budget_remaining -= upgrade_cost_bytes
                upgraded_any = True
                
        # If we looped through all tensors and couldn't afford to upgrade ANY of them,
        # we have hit the packing limit of our budget.
        if not upgraded_any:
            break
            
    return allocations
