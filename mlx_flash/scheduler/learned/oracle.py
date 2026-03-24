import numpy as np
from typing import Dict, Any, List
from .rls import OnlineRLS

class LearnedOracle:
    """
    Replaces the heuristic Belady cost model with an online-learning system.
    Predicts precise latency and reuse probability based on continuous profiler feedback.
    """
    def __init__(self, total_layers: int = 32):
        self.total_layers = total_layers
        
        # IO Latency Model
        # Features: [1.0 (bias), size_mb, is_moe_expert, queue_depth]
        self.io_model = OnlineRLS(num_features=4, forgetting_factor=0.99)
        
        # We pre-warm the RLS weights with reasonable Apple Silicon defaults
        # Assume ~3.5 GB/s -> ~0.28 ms per MB
        self.io_model.w = np.array([0.0, 0.28, 0.0, 0.5]) 
        
        # Compute Latency Model
        # Features: [1.0 (bias), size_mb, is_moe_expert]
        self.compute_model = OnlineRLS(num_features=3, forgetting_factor=0.99)
        self.compute_model.w = np.array([0.0, 0.1, 0.0]) # Rough guess: compute is faster than IO
        
        # Reuse Probability Model (LinUCB-style tracker for caching)
        # Features: [1.0 (bias), relative_depth (0..1), router_prob (0..1)]
        self.reuse_model = OnlineRLS(num_features=3, forgetting_factor=0.95) # Forgets faster because topic changes

    def extract_io_features(self, task: Any, queue_depth: int) -> np.ndarray:
        # Assuming task is a ResourceTask, and we can infer size. 
        # In a real system, the task object must carry byte size.
        size_mb = getattr(task, 'size_bytes', 16 * 1024 * 1024) / 1e6
        is_moe = 1.0 if 'expert' in task.task_type else 0.0
        return np.array([1.0, size_mb, is_moe, float(queue_depth)])

    def extract_reuse_features(self, layer_idx: int, router_prob: float) -> np.ndarray:
        rel_depth = layer_idx / max(1, self.total_layers)
        return np.array([1.0, rel_depth, router_prob])

    def get_urgency_score(self, current_layer: int, target_layer: int, 
                          is_blocking: bool, task_type: str, freq: float = 1.0, 
                          task_size_bytes: int = 16*1024**2, queue_depth: int = 0) -> float:
        """
        Replaces the hardcoded `UnifiedCostModel.get_urgency_score`.
        """
        # Distance to execution (wrap around for cyclic generation)
        distance = target_layer - current_layer
        if distance < 0: distance += self.total_layers
        
        if distance == 0 and task_type.startswith('io_read'):
            return float('inf') # Hard deadline
            
        # Predict how long this will take
        size_mb = task_size_bytes / 1e6
        is_moe = 1.0 if 'expert' in task_type else 0.0
        
        x_io = np.array([1.0, size_mb, is_moe, float(queue_depth)])
        predicted_ms = self.io_model.predict(x_io)
        
        # Base urgency driven by how much time we need vs how much time we have
        # (Using distance as a proxy for time-we-have).
        base_urgency = (predicted_ms * 100.0) / max(1, distance)
        
        # If MoE expert, multiply by learned reuse/routing probability
        if is_moe:
            x_reuse = self.extract_reuse_features(target_layer, freq)
            p_reuse = self.reuse_model.predict(x_reuse)
            # Clip between 0 and 1
            p_reuse = max(0.0, min(1.0, p_reuse))
            base_urgency *= p_reuse
            
        return base_urgency

    def observe_and_train(self, profiler_data: Dict[str, Any]):
        """
        Called at the end of a token generation pass. 
        Ingests the real-world millisecond timings recorded by the StreamingProfiler
        and updates the RLS weights instantly.
        """
        io_records = profiler_data.get('io_tasks', [])
        for record in io_records:
            # record expected shape: {'size_bytes': int, 'is_moe': bool, 'queue_depth': int, 'duration_ms': float}
            x = np.array([
                1.0, 
                record['size_bytes'] / 1e6, 
                1.0 if record['is_moe'] else 0.0, 
                float(record['queue_depth'])
            ])
            y_true = record['duration_ms']
            self.io_model.update(x, y_true)
            
        cache_records = profiler_data.get('cache_evictions', [])
        for record in cache_records:
            # record expected shape: {'layer_idx': int, 'router_prob': float, 'reused_soon': bool}
            x = self.extract_reuse_features(record['layer_idx'], record.get('router_prob', 1.0))
            y_true = 1.0 if record['reused_soon'] else 0.0
            self.reuse_model.update(x, y_true)
