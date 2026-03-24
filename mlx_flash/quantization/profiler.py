import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Any

class ActivationVarianceProfiler:
    """
    Profiles a model to determine the sensitivity of its linear layers to quantization noise.
    Uses the magnitude/variance of the input activations as a proxy for sensitivity.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.activation_stats: Dict[str, float] = {}
        self._hooks: List[Any] = []
        
    def _create_hook(self, name: str):
        def hook(module: nn.Module, x: mx.array, output: mx.array):
            # Calculate the L2 norm of the input activations across the sequence
            # S_l = 1/N sum(||X_l||_2)
            
            # x is typically a tuple of arguments, we take the first one (the main hidden state)
            inputs = x[0] if isinstance(x, tuple) else x
            
            # Normalize by number of elements to get a stable score
            magnitude = mx.sum(mx.square(inputs)).item() / inputs.size
            
            if name in self.activation_stats:
                # Exponential moving average if called multiple times (e.g. over a dataset)
                self.activation_stats[name] = 0.9 * self.activation_stats[name] + 0.1 * magnitude
            else:
                self.activation_stats[name] = magnitude
                
        return hook

    def attach_hooks(self):
        """Attaches forward hooks to all nn.Linear layers in the model."""
        self.remove_hooks() # Ensure clean state
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # MLX doesn't have a built-in `register_forward_hook` like PyTorch.
                # In a real MLX implementation, you have to patch the __call__ method manually.
                original_call = module.__call__
                
                def patched_call(x, *args, _name=name, _orig=original_call, **kwargs):
                    out = _orig(x, *args, **kwargs)
                    
                    # Compute sensitivity metric inline
                    magnitude = mx.sum(mx.square(x)).item() / x.size
                    if _name in self.activation_stats:
                        self.activation_stats[_name] = 0.9 * self.activation_stats[_name] + 0.1 * magnitude
                    else:
                        self.activation_stats[_name] = magnitude
                        
                    return out
                
                module.__call__ = patched_call
                self._hooks.append((module, original_call))

    def remove_hooks(self):
        """Restores the original __call__ methods."""
        for module, original_call in self._hooks:
            module.__call__ = original_call
        self._hooks.clear()

    def profile_dataset(self, calibration_data: List[mx.array]):
        """Runs the calibration dataset through the model to gather statistics."""
        self.attach_hooks()
        
        try:
            for batch in calibration_data:
                # We don't need gradients or the actual output, just the forward pass
                # to trigger the patched __call__ methods.
                _ = self.model(batch)
                mx.eval(self.activation_stats) # Ensure evaluation happens
        finally:
            self.remove_hooks()
            
        return self.activation_stats

    def get_tensor_metadata(self) -> List[Dict[str, Any]]:
        """
        Combines the model's structural metadata (shapes) with the calculated 
        sensitivity scores to feed into the bit allocator.
        """
        tensors = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight_shape = module.weight.shape
                # If a layer wasn't profiled (e.g. didn't execute), assign 0 sensitivity
                sensitivity = self.activation_stats.get(name, 0.0) 
                
                # We specifically name the weight tensor to match safetensors convention
                tensor_name = f"{name}.weight"
                
                # Heuristics: artificially boost highly critical architectural points
                if "o_proj" in name:
                     sensitivity *= 2.0 # Attention output is notoriously sensitive
                elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                     sensitivity *= 0.5 # MLPs are very robust to quantization
                     
                tensors.append({
                    'name': tensor_name,
                    'shape': weight_shape,
                    'sensitivity': sensitivity
                })
                
        return tensors
