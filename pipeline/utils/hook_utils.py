
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import pdb

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

import h5py
import torch
from torch import Tensor
from pathlib import Path
from jaxtyping import Float

class HDF5ProjectionSaver:
    def __init__(self, filepath: str = "./projections.h5", max_samples: int = 1000000, buffer_size: int = 5000):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        self.file = h5py.File(filepath, 'w')
        self.dataset = None
        self.current_idx = 0
        self.max_samples = max_samples
        self.metadata = self.file.create_group('metadata')
        
        # Buffer for batching writes
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self, proj_v: Tensor):
        """Add projections - flattens batch and sequence dimensions"""
        # proj_v shape: [batch_size, seq_len, d_model]
        # Flatten to: [batch_size * seq_len, d_model]
        proj_cpu = proj_v.detach().cpu()
        flattened = proj_cpu.view(-1, proj_cpu.shape[-1])  # [num_tokens, d_model]
        
        self.buffer.append(flattened)
        
        # Flush buffer when it reaches buffer_size
        if sum(b.shape[0] for b in self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffer to disk"""
        if not self.buffer:
            return
        
        # Concatenate all buffered projections
        proj_np = torch.cat(self.buffer, dim=0).numpy()  # [num_tokens, d_model]
        
        # Initialize dataset on first flush
        if self.dataset is None:
            d_model = proj_np.shape[-1]
            self.dataset = self.file.create_dataset(
                'projections',
                shape=(self.max_samples, d_model),
                dtype='float16',
                chunks=(1000, d_model),
                compression='gzip',
                compression_opts=4
            )
            self.metadata.attrs['d_model'] = d_model
        
        batch_size = proj_np.shape[0]
        end_idx = self.current_idx + batch_size
        
        if end_idx > self.max_samples:
            self.dataset.resize(end_idx + 10000, axis=0)
            self.max_samples = end_idx + 10000
        
        # Single write operation
        self.dataset[self.current_idx:end_idx] = proj_np
        self.current_idx = end_idx
        
        # Clear buffer
        self.buffer.clear()
    
    def get_total_samples(self):
        """Get total samples including unflushed buffer"""
        flushed_count = self.current_idx
        buffered_count = sum(b.shape[0] for b in self.buffer)
        return flushed_count + buffered_count
    
    def close(self):
        """Flush remaining buffer and close file"""
        self._flush_buffer()
        
        if self.dataset is not None and self.current_idx < len(self.dataset):
            self.dataset.resize(self.current_idx, axis=0)
            self.metadata.attrs['total_samples'] = self.current_idx
        
        total_saved = self.current_idx
        self.file.close()
        #print(f"Saved {total_saved} token projections to {self.filepath}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Loading is now trivial:
def load_projections(filepath: str):
    """Load all token projections"""
    with h5py.File(filepath, 'r') as f:
        projections = torch.from_numpy(f['projections'][:])  # [num_tokens, d_model]
        print(f"Loaded {projections.shape[0]} token projections with d_model={projections.shape[1]}")
    return projections

# Or load in batches:
def analyze_projections_batched(filepath: str, batch_size: int = 10000):
    with h5py.File(filepath, 'r') as f:
        total = f['projections'].shape[0]
        for i in range(0, total, batch_size):
            batch = torch.from_numpy(f['projections'][i:i+batch_size])
            # Do analysis on batch
            print(f"Processing batch {i//batch_size + 1}, shape: {batch.shape}")

def get_save_direction_input_pre_hook(
    direction: Tensor, 
    reference: Tensor | None = None, 
    coeff: Tensor | None = None,
    projection_saver: HDF5ProjectionSaver | None = None
):
    def hook_fn(module, input):
        nonlocal direction, reference, coeff, projection_saver

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()

        if reference is None:
            reference = torch.zeros_like(activation[0])
        if coeff is None:
            coeff = torch.tensor([0])
        
        direction = direction.to(activation)
        reference = reference.to(activation)
        coeff = coeff.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Save projection to HDF5
        #if projection_saver is not None:

        projection_saver.add(proj_v)

        # Don't modify activation (observation only)
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    
    return hook_fn


def caa_add_hook(direction: Tensor, multiplier = 1):
    def hook_fn(module, input):
        nonlocal direction, multiplier
 
        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()
        
        direction = direction.to(activation)

        activation = activation + multiplier * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_affine_direction_ablation_input_pre_hook(direction: Tensor, 
                                          reference: Tensor | None = None, 
                                          coeff: Tensor| None = None):
    def hook_fn(module, input):
        nonlocal direction, reference, coeff

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()

        if reference is None:
            reference = torch.zeros_like(activation[0])
        if coeff is None:
            coeff = torch.Tensor([0])
        
        direction = direction.to(activation)
        reference = reference.to(activation)
        coeff = coeff.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Apply affine transformation
        activation = activation - proj_v + proj_ref + coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_affine_direction_ablation_output_hook(direction: Tensor, 
                                        reference: Tensor | None = None):
    def hook_fn(module, input, output):
        nonlocal direction, reference

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output.clone()

        if reference is None:
            reference = torch.zeros_like(activation[0])

        direction = direction.to(activation)
        reference = reference.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Apply affine transformation
        activation = activation - proj_v + proj_ref + direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_affine_activation_addition_input_pre_hook(
    direction: Float[Tensor, "d_model"], 
    coeff: Float[Tensor, ""], 
    reference: Float[Tensor, "d_model"] | None = None
):
    def hook_fn(module, input):
        nonlocal direction, reference, coeff

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()

        # Default reference to zero tensor if not provided
        if reference is None:
            reference = torch.zeros_like(activation[0])

        direction = direction.to(activation)
        reference = reference.to(activation)
        coeff = coeff.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)


        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Apply affine transformation
        modified_activation = activation - proj_v + proj_ref + coeff * direction

        # Return modified activation
        if isinstance(input, tuple):
            return (modified_activation, *input[1:])
        else:
            return modified_activation

    return hook_fn


def get_affine_activation_addition_input_post_hook(
    direction: Float[Tensor, "d_model"], 
    coeff: Float[Tensor, ""], 
    reference: Float[Tensor, "d_model"] | None = None
):
    def hook_fn(module, input, output):
        nonlocal direction, reference, coeff

        if reference is None:
            reference = torch.zeros_like(activation[0])
            

        if isinstance(output, tuple):
            activation = output[0]
            other_outputs = output[1:]
            
            direction = direction.to(activation)
            reference = reference.to(activation)
            coeff = coeff.to(activation)
            
            # Normalize the direction vector
            normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

            # Compute projections
            proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
            proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

            # Apply affine transformation
            modified_activation = activation - proj_v + proj_ref + coeff * direction
            
            return (modified_activation,) + other_outputs
        else:
            activation = output

            direction = direction.to(activation)
            reference = reference.to(activation)
            coeff = coeff.to(activation)
            
            # Normalize the direction vector
            normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

            # Compute projections
            proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
            proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

            # Apply affine transformation
            modified_activation = activation - proj_v + proj_ref + coeff * direction
            return modified_activation
    return hook_fn

    
def get_linear_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple) and len(input) > 0:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        d = direction.to(activation)
        d = d / (d.norm() + 1e-8)

        activation -= (activation @ d).unsqueeze(-1) * d 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_linear_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction
     
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output
        d = direction.to(activation)
        d = d / (d.norm() + 1e-8)
 
        activation -= (activation @ d).unsqueeze(-1) * d 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_linear_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_linear_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_linear_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_linear_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks


def get_linear_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: float):
    def hook_fn(module, input):
        nonlocal vector, coeff

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn