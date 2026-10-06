"""
steer_components.py
======================
Stripped steering pipeline that hooks into MLP and attention components
directly rather than full transformer blocks.

Usage:
    python -u -m pipeline.steer_components --model_path microsoft/Phi-4-mini-instruct
         --anchor_layer 14 --use_existing
"""

import os
import gc
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns 

import torch
from tqdm import tqdm
import ctypes
import time

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions_repit import generate_category_directions
from pipeline.submodules.completions_helper import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset,
)
from pipeline.utils.hook_utils import *
from pipeline.submodules.evaluate_jailbreak import unload_llamaguard_model
from dataset.load_dataset import load_dataset_split

from pipeline.sweep_layers import get_norm_explained_mask


# ── Direction helpers ──────────────────────────────────────────────────────────

def get_norm_explained_mask(direction: torch.Tensor, energy_threshold: float = 0.9):
    sorted_indices = direction.abs().argsort(descending=True)
    squared        = direction[sorted_indices] ** 2
    cumulative     = squared.cumsum(0) / squared.sum()
    k              = int((cumulative < energy_threshold).sum().item()) + 1
    return sorted_indices[:k]


class ProjectionCollector:
    def __init__(self):
        self.projections = {}
    
    def add_projection(self, layer_name, projection_magnitude):
        if layer_name not in self.projections:
            self.projections[layer_name] = []
        # Convert scalar to 1D tensor before appending
        self.projections[layer_name].append(projection_magnitude.detach().cpu().unsqueeze(0))
    
    def get_means(self):
        return {name: torch.cat(projs).mean().item() 
                for name, projs in self.projections.items()}

def get_norm_explained_mask(direction, energy_threshold=0.9):
    """
    Return indices of the fewest coordinates that explain `energy_threshold`
    fraction of the direction's squared L2 norm.
    """
    sorted_indices = direction.abs().argsort(descending=True)
    squared        = direction[sorted_indices] ** 2
    cumulative     = squared.cumsum(0) / squared.sum()
    k              = int((cumulative < energy_threshold).sum().item()) + 1
    return sorted_indices[:k]

def make_sparse_projection_output_hook(direction, mask_indices):
    """
    Linear orthogonalization of the refusal direction, restricted to a sparse
    coordinate mask. Coordinates outside the mask are not projected.
    """
    def hook_fn(module, input, output):
        nonlocal direction, mask_indices
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output

        activation = output[0] if isinstance(output, tuple) else output
        dtype, device = activation.dtype, activation.device
        d = direction.to(device=device, dtype=dtype)
        norm_d   = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        proj_v   = (activation @ norm_d).unsqueeze(-1) * norm_d
        mask = torch.zeros(d.shape[0], dtype=torch.bool, device=device)
        mask[mask_indices] = True
        proj_v[..., ~mask]   = 0.0
        activation_mod = activation - proj_v
        return (activation_mod, *output[1:]) if isinstance(output, tuple) else activation_mod
    return hook_fn

def get_projection_measurement_hook(direction: Tensor, layer_name: str, collector: ProjectionCollector):
    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        
        d = direction.to(activation.device, activation.dtype)
        d = d / (d.norm() + 1e-8)
        
        # Project onto direction and measure magnitude
        projections = activation @ d  # [batch, seq]
        proj_magnitude = projections.abs().mean()  # mean over batch and sequence
        
        collector.add_projection(layer_name, proj_magnitude)
        
        # Don't modify input
        return input
    
    return hook_fn

def rebuild_module_set(prev_set, model_base, directions, anchor_layer, energy_threshold=-1, complement=False):
    """
    Rebuild live (module, hook_fn) tuples from a list of component names or
    (name, ...) tuples after a model reload.

    Parameters
    ----------
    prev_set          : list of names or (name, module, hook) triples
    model_base        : freshly constructed model base
    directions        : dict of layer_idx -> direction tensor
    anchor_layer      : which layer's direction to use
    energy_threshold  : fraction of squared L2 norm to retain when auto-computing
                        the sparse mask (default 0.9).  Set to 1.0 to use dense hooks.
    complement        : if True, build the complement set instead — i.e. all
                        components NOT in prev_set that are strictly or in
                        anchor_layer (embed + attn_0..anchor_layer-1 +
                        mlp_0..anchor_layer).
    """
    direction = directions[anchor_layer]

    # ── Resolve sparse mask ────────────────────────────────────────────────────
    if energy_threshold < 1.0 and energy_threshold > 0:
        mask = get_norm_explained_mask(direction, energy_threshold=energy_threshold)

        def make_linear_hook(dir):
            return make_sparse_projection_output_hook(dir, mask)
        def make_embed_hook(dir):
            return make_embedding_hook(dir, mask_indices=mask)
    else:
        def make_linear_hook(dir):
            return get_linear_direction_ablation_output_hook(dir)
        def make_embed_hook(dir):
            return make_embedding_hook(dir, mask_indices=None)

    # ── Unwrap model ───────────────────────────────────────────────────────────
    model = model_base.model
    if hasattr(model, "language_model"):
        model = model.language_model

    if complement:
        # Build the full set (embed + all attn/mlp)
        # then subtract whatever is already in prev_set.
        names_in_set = {
            (entry[0] if isinstance(entry, tuple) else entry)
            for entry in prev_set
        }

        # get_input_embeddings() resolves the text embedding across all wrappers,
        # incl. Mistral3 (model.model.language_model.embed_tokens, 3 levels deep).
        embed_module = model_base.model.get_input_embeddings()
        upstream_blocks = [("embed", embed_module, make_embed_hook(direction))]
        for i in range(anchor_layer):
            upstream_blocks.append((f"attn_{i}", model_base.model_attn_modules[i], make_linear_hook(direction)))
            upstream_blocks.append((f"mlp_{i}",  model_base.model_mlp_modules[i],  make_linear_hook(direction)))

        module_set = [(name, mod, hook) for name, mod, hook in upstream_blocks
                      if name not in names_in_set]

        mode = f"sparse (energy_threshold={energy_threshold})" if energy_threshold != -1 else "dense"
        print(f"Rebuilt COMPLEMENT circuit with {len(module_set)} components [{mode}] "
              f"({len(names_in_set)} excluded) from NEW model base")
        return module_set

    # ── Normal (non-complement) path ───────────────────────────────────────────
    module_set = []
    for entry in prev_set:
        name = entry[0] if isinstance(entry, tuple) else entry

        if name == "embed":
            embed_module = model_base.model.get_input_embeddings()
            hook_fn = make_embed_hook(direction)
            module_set.append((name, embed_module, hook_fn))

        elif name.startswith("mlp_"):
            idx     = int(name.split("_")[1])
            module  = model_base.model_mlp_modules[idx]
            hook_fn = make_linear_hook(direction)
            module_set.append((name, module, hook_fn))

        elif name.startswith("attn_"):
            idx     = int(name.split("_")[1])
            module  = model_base.model_attn_modules[idx]
            hook_fn = make_linear_hook(direction)
            module_set.append((name, module, hook_fn))

        else:
            print(f"rebuild_module_set: unrecognised component '{name}', skipping.")

    mode = f"sparse (energy_threshold={energy_threshold})" if energy_threshold != -1 else "dense"
    print(f"Rebuilt circuit with {len(module_set)} components [{mode}] from NEW model base")
    return module_set

def create_short_tag(circuit):
    component_names = sorted([comp[0] for comp in circuit])
    # Convert mlp_3 -> m3, attn_5 -> a5, embed -> e
    short_names = []
    for name in component_names:
        if name.startswith("mlp_"):
            short_names.append("m" + name.split("_")[1])
        elif name.startswith("attn_"):
            short_names.append("a" + name.split("_")[1])
        elif name == "embed":
            short_names.append("e")
    return "".join(short_names)

def get_asr_from_evaluation_file(cfg, tag):
    """Helper to extract ASR from evaluation JSON"""
    eval_path = os.path.join(cfg.artifact_path(), "completions/pruning", f"harmful_{tag}_evaluations.json")
    with open(eval_path) as f:
        data = json.load(f)
    return data.get("llamaguard3_success_rate", data.get("success_rate", 0.0))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)
        
def heuristic_elimination_search(cfg, model_base, directions, harmful_val, harmless_mean, anchor_layer, heuristic_threshold = 0.95,
                               temperature = None, batch_size=128, use_existing=False):
    """
    Perform greedy feature elimination on MLP/Attention/embedding components
    using fraction-based thresholds.
    """

    filename = f"heuristic_elimination_anchor_{anchor_layer}"
    if temperature:
        filename += f"_T{temperature}"
    if heuristic_threshold:
        filename += f"_threshold{heuristic_threshold}"

    save_path = os.path.join(cfg.artifact_path(), f"{filename}.json")
    
    # --- Load existing set if requested ---
    if use_existing and os.path.exists(save_path):
        print(f"Loading existing set from {save_path}")
        with open(save_path, "r") as f:
            saved_data = json.load(f)
    
        component_data = saved_data.get("components")
        
        minimal_set = []
        model = model_base.model
        
        for name in component_data:
            # Rebuild module reference
                
            if name.startswith("embed"):
                embed_module = model_base.model.get_input_embeddings()
                hook_fn = make_embedding_hook(directions[anchor_layer])
            elif name.startswith("mlp_"):
                idx = int(name.split("_")[1])
                module = model_base.model_mlp_modules[idx]
                hook_fn = get_linear_direction_ablation_output_hook(directions[anchor_layer])
            elif name.startswith("attn_"):
                idx = int(name.split("_")[1])
                module = model_base.model_attn_modules[idx]
                hook_fn = get_linear_direction_ablation_output_hook(directions[anchor_layer])
            else:
                continue
            minimal_set.append((name, module, hook_fn))
        print(f"Loaded {len(minimal_set)} components")
        return minimal_set

    # --- Otherwise, compute set ---
    direction = directions[anchor_layer]
    hook_fn = get_linear_direction_ablation_output_hook(direction)
    
    
    # Build all components up to anchor layer only
    all_components = []
    for i in range(anchor_layer):
        all_components.append((f"attn_{i}", model_base.model_attn_modules[i], hook_fn))
        all_components.append((f"mlp_{i}", model_base.model_mlp_modules[i], hook_fn))
    
    embed_module = model_base.model.get_input_embeddings()
    embed_hook_fn = make_embedding_hook(direction)
    all_components.append(("embed", embed_module, embed_hook_fn))
    
    # Get actual unsteered baseline by running the model with no steering hooks
    def get_unsteered_baseline():
        representations = []
        
        def collect_hook(module, input, output):
            if isinstance(input, tuple):
                activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
            else:
                activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()
            representations.append(activation.mean(dim=1).detach())
        
        hook = model_base.model_block_modules[anchor_layer].register_forward_pre_hook(collect_hook)
        
        for batch_start in range(0, min(len(harmful_val), 256), batch_size):
            batch = harmful_val[batch_start:batch_start+batch_size]
            prompts = [d["instruction"] if not isinstance(d, str) else d for d in batch]
            
            inputs = model_base.tokenizer(prompts, return_tensors='pt', padding=True,
                                        truncation=True, max_length=512).to(model_base.model.device)
            with torch.no_grad():
                model_base.model(**inputs)
        
        for hook in hooks:
            hook.remove()
        del hooks, collector, measure_hook_fn
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return torch.cat(representations, dim=0).mean(dim=0)

    # --- Helper to test a subset ---
    def test_component_subset(component_subset):
        collector = ProjectionCollector()
        hooks = []

        # Hook selected components
        for name, module, hook_function in component_subset:
            hooks.append(module.register_forward_hook(hook_function))
        
        # Anchor layer measurement hook
        measure_hook_fn = get_projection_measurement_hook(direction, "test", collector)
        measure_hook = model_base.model_block_modules[anchor_layer].register_forward_pre_hook(measure_hook_fn)
        hooks.append(measure_hook)
        
        # Forward pass
        for batch_start in range(0, min(len(harmful_val), 256), batch_size):
            batch = harmful_val[batch_start:batch_start+batch_size]
            prompts = [d["instruction"] if not isinstance(d, str) else d for d in batch]
            
            inputs = model_base.tokenizer(prompts, return_tensors='pt', padding=True,
                                          truncation=True, max_length=512).to(model_base.model.device)
            with torch.no_grad():
                model_base.model(**inputs)
        
        projection = collector.get_means().get("test", float('inf'))
        
        for hook in hooks:
            hook.remove()
        
        return projection
    
    # Get baselines using original working approach  
    unaltered_projection = test_component_subset([])
    full_projection = test_component_subset(all_components)
    total_improvement = unaltered_projection - full_projection
    
    current_set = all_components.copy()
    
    print(f"Starting with all {len(current_set)} components: projection={full_projection:.4f}")
    print(f"Without steering: unaltered projection={unaltered_projection:.4f}")
    print(f"Total improvement possible: {total_improvement:.4f}")
    
    # Stop when we've lost more than 5% of the total improvement
    threshold_fraction = heuristic_threshold  # Keep 95% of improvement
    min_improvement_required = threshold_fraction * total_improvement
    threshold_projection = unaltered_projection - min_improvement_required
    
    print(f"Will stop when projection > {threshold_projection:.4f} ({threshold_fraction:.1%} of target)")
    
    while len(current_set) > 1:
        iteration_projection = test_component_subset(current_set)
        
        if iteration_projection > threshold_projection:
            print(f"Reached threshold: {iteration_projection:.4f} > {threshold_projection:.4f}")
            break

        removal_impacts = []
        for i, component in enumerate(current_set):
            test_set = current_set[:i] + current_set[i+1:]
            projection = test_component_subset(test_set)
            impact = abs(projection - iteration_projection)
            removal_impacts.append((i, component, projection, impact))
            print(f"  Try removing {component[0]:10s}: projection={projection:.4f}, impact={impact:.6f}")
        
  
        if temperature:
            impacts = [x[3] for x in removal_impacts]  
            probs = softmax([-impact/temperature for impact in impacts])
            removal_idx = np.random.choice(len(removal_impacts), p=probs)
            i, component, projection, impact = removal_impacts[removal_idx]
        else:
            best_removal = min(removal_impacts, key=lambda x: x[2])
            i, component, projection, impact = best_removal
        
        # Check if removing this component would violate our threshold
        if projection > threshold_projection:
            print(f"  → STOPPING: removing {component[0]} would exceed threshold ({projection:.4f} > {threshold_projection:.4f})")
            break
            
        removed = current_set.pop(i)
        iteration_projection = projection
        improvement_current = unaltered_projection - iteration_projection
        fraction_current = improvement_current / total_improvement
        print(f"  → REMOVED {removed[0]}: projection={iteration_projection:.4f} ({fraction_current:.1%} of target), {len(current_set)} components left")
    
    final_improvement = unaltered_projection - iteration_projection
    final_fraction = final_improvement / total_improvement
    
    print(f"\nFinal elimination circuit: {len(current_set)} components")
    print(f"Achieved {final_fraction:.1%} of target improvement")
    for name, module, _ in current_set:
        print(f"  {name}")
    
    # --- Save set with metrics ---
    os.makedirs(cfg.artifact_path(), exist_ok=True)
    save_data = {
        "anchor_layer": anchor_layer,
        "components": [name for name, _, _ in current_set],
        "final_projection": iteration_projection,
        "unaltered_projection": unaltered_projection,
        "full_projection": full_projection,
        "improvement_achieved": final_improvement,
        "improvement_possible": total_improvement,
        "fraction_of_target": final_fraction
    }
    
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Greedy set saved to {save_path}")
    
    return current_set

def iterative_pruning(cfg, model_base, minimal_set, directions, harmful_val, anchor_layer, 
                        batch_size=128, threshold=0.05, use_existing = False, save_suffix=""):
    """
    Generate all completions first, then evaluate in batch to avoid model loading/unloading
    """
    current_set = minimal_set.copy()
    
    save_path = os.path.join(cfg.artifact_path(), f"exhaustive_set_anchor_{anchor_layer}{save_suffix}.json")
    
    # --- Load existing set if requested ---
    if use_existing and os.path.exists(save_path):
        print(f"Loading existing set from {save_path}")
        with open(save_path, "r") as f:
            saved_data = json.load(f)
        
        minimal_set = []
        for entry in saved_data["components"]:
            name = entry
            # Rebuild module reference
            if name == "embed":
                module = model_base.model.get_input_embeddings()
                hook_fn = make_embedding_hook(directions[anchor_layer])
            elif name.startswith("mlp_"):
                idx = int(name.split("_")[1])
                module = model_base.model_mlp_modules[idx]
                hook_fn = get_linear_direction_ablation_output_hook(directions[anchor_layer])
            elif name.startswith("attn_"):
                idx = int(name.split("_")[1])
                module = model_base.model_attn_modules[idx]
                hook_fn = get_linear_direction_ablation_output_hook(directions[anchor_layer])
            else:
                continue
            minimal_set.append((name, module, hook_fn))
        print(f"Loaded {len(minimal_set)} components")
        return minimal_set
    # First, generate baseline with current full set
    baseline_tag = run_component_test(
        cfg, model_base, current_set, directions,
        ablation="baseline_full",
        anchor_layer=anchor_layer,
        harmful_prompts=harmful_val,
        batch_size=batch_size,
        use_existing=True
    )

     # Now unload model and evaluate all in batch
    print("Unloading model and evaluating all completions...")
    unload_model(model_base)
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate all tags
    evaluate_tags(cfg, baseline_tag, batch_size=batch_size, use_existing=True)

    # Get ASRs and find best removal
    baseline_asr = get_asr_from_evaluation_file(cfg, os.path.basename(baseline_tag[0][0]))

    if baseline_asr < 0.5:
        print(f"→ STOPPING: underperformant circuit")
        # --- Save set ---
        os.makedirs(cfg.artifact_path(), exist_ok=True)
        asr_data = [{"component" : name, "drop in ASR": -1, 
                    "new ASR": -1} for name, _, _ in current_set]
        asr_save_path = os.path.join(cfg.artifact_path(), f"final_exhaustive_ASRs{save_suffix}.json")
        with open(asr_save_path, "w") as f:
            json.dump(asr_data, f, indent=2)
        print(f"Failed set saved to {asr_save_path}")
        
        # Reload model for next iteration
        model_base = construct_model_base(cfg.model_path)
        current_set = rebuild_module_set(current_set, model_base, directions, anchor_layer)
        return current_set   

    model_base = construct_model_base(cfg.model_path)
    current_set = rebuild_module_set(current_set, model_base, directions, anchor_layer)
    removal_candidates = []
    
    first_iter = True
    while len(current_set) > 1:
        print(f"\nGenerating completions for {len(current_set)} removal tests...")

        # Generate completions for all possible removals
        generation_tags = []
        for i, (name, module, hook) in enumerate(current_set):
            test_set = current_set[:i] + current_set[i+1:]
            
            tags = run_component_test(
                cfg, model_base, test_set, directions,
                ablation=f"remove_{name}",
                anchor_layer=anchor_layer,
                harmful_prompts=harmful_val,
                batch_size=batch_size,
                use_existing=True
            )
            generation_tags.extend(tags)

        # Now unload model and evaluate all in batch
        print("Unloading model and evaluating all completions...")
        unload_model(model_base)
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate all tags
        all_tags = baseline_tag + generation_tags
        evaluate_tags(cfg, all_tags, batch_size=batch_size, use_existing=True)

        # Get ASRs and find best removal
        baseline_asr = get_asr_from_evaluation_file(cfg, os.path.basename(baseline_tag[0][0]))

        removal_candidates = []


        for i, entry in enumerate(generation_tags):
            tagname, path = entry
            removed_component = tagname.split("remove_")[1].split("_eval")[0]
            asr = get_asr_from_evaluation_file(cfg, tagname)
            drop = baseline_asr - asr
            removal_candidates.append((removed_component, drop, asr, i))
            print(f"  Remove {removed_component:10s}: ASR={asr:.3f}, drop={drop:.3f}")

        # Remove best candidate or stop
        best_removal = min(removal_candidates, key=lambda x: x[1])
        removed_component, drop, new_asr, idx = best_removal

        if first_iter:
            os.makedirs(cfg.artifact_path(), exist_ok=True)
            asr_data = [{"component" : removed_component, "drop in ASR": drop, 
                        "new ASR": new_asr} for removed_component, drop, new_asr, _ in removal_candidates]
            first_asr_save_path = os.path.join(cfg.artifact_path(), f"first_exhaustive_ASRs{save_suffix}.json")
            with open(first_asr_save_path, "w") as f:
                json.dump(asr_data, f, indent=2)
            first_iter = False
            print(f"First iteration set saved to {first_asr_save_path}")

        if drop <= threshold:
            current_set.pop(idx)
            print(f"→ REMOVED {removed_component} (drop: {drop:.3f}), {len(current_set)} left")
            
            # Reload model for next iteration
            model_base = construct_model_base(cfg.model_path)
            current_set = rebuild_module_set(current_set, model_base, directions, anchor_layer)
        else:
            print(f"→ STOPPING: drop {drop:.3f} > threshold {threshold}")
            # --- Save set ---
            os.makedirs(cfg.artifact_path(), exist_ok=True)
            asr_data = [{"component" : removed_component, "drop in ASR": drop, 
                        "new ASR": new_asr} for removed_component, drop, new_asr, _ in removal_candidates]
            asr_save_path = os.path.join(cfg.artifact_path(), f"final_exhaustive_ASRs{save_suffix}.json")
            with open(asr_save_path, "w") as f:
                json.dump(asr_data, f, indent=2)
            print(f"Pruned set ASRs saved to {asr_save_path}")
            
            # Reload model for next iteration
            model_base = construct_model_base(cfg.model_path)
            current_set = rebuild_module_set(current_set, model_base, directions, anchor_layer)
            break

    # --- Save set ---
    save_data = {"components": [name for name, _, _ in current_set]}
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Pruned set saved to {save_path}")
    
    return current_set

def analyze_minimal_set_svd(minimal_set, model_base, directions, anchor_layer, cfg, save_suffix ="", use_existing = False):
    """
    For each module in minimal_set (MLP/Attn), run SVD on its output weight (residual space)
    and measure alignment of anchor-layer direction with the output eigenspace.
    """
    anchor_dir = directions[anchor_layer].to(torch.float32)
    anchor_dir = anchor_dir / (anchor_dir.norm() + 1e-8)

    save_path = os.path.join(cfg.artifact_path(), f"minimal_set_svd_residual_anchor_{anchor_layer}{save_suffix}.json")
    if os.path.exists(save_path) and use_existing:
        with open(save_path) as f:
            svd_results = json.load(f)
        return svd_results

    svd_results = {}

    for name, module, hook in minimal_set:
        if name.lower().startswith("embedding"):
            continue

        W = None
        if name.startswith("mlp_"):
            if hasattr(module, "down_proj"):
                W = module.down_proj.weight.data  # [out_features, hidden_features]
        elif name.startswith("attn_"):
            if hasattr(module, "o_proj"):
                W = module.o_proj.weight.data  # [residual, heads*head_dim]

        if W is None:
            print(f"Skipping {name}, no recognized output weight")
            continue

        W_cpu = W.detach().cpu().float()
        out_dim = W_cpu.shape[0]

        # Make sure anchor_dir matches output/residual dimension
        if anchor_dir.shape[0] != out_dim:
            print(f"Skipping {name}, anchor_dir dim {anchor_dir.shape[0]} != output dim {out_dim}")
            continue

        # SVD: W = U S V^T → U spans output/residual space
        try:
            U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
            U = U.to(anchor_dir)
        except RuntimeError as e:
            print(f"SVD failed for {name}: {e}")
            continue

        # Check alignment with TOP singular vectors (strongest writing directions)
        top_k = 50  # Check top 50 singular vectors
        top_U = U[:, :top_k]  # Top k columns of U

        # How much of anchor_dir is captured by top k writing directions?
        proj_top = top_U @ (top_U.T @ anchor_dir)
        alignment_with_top_k = (proj_top @ anchor_dir).item() / (anchor_dir.norm().item() ** 2)

        print(f"{name}: {alignment_with_top_k:.4f} alignment with top-{top_k} singular vectors")

        svd_results[name] = {
            f"alignment with top{top_k} singular vecs": alignment_with_top_k,
        }

    # Write results
    os.makedirs(cfg.artifact_path(), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(svd_results, f, indent=2)

    print(f"SVD residual-space analysis written to {save_path}")
    return svd_results


def build_directions(cfg, category_directions, num_layers, pos=-1):
    dir_path          = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(dir_path, exist_ok=True)
    refusal_directions_path    = os.path.join(dir_path, "refusal_directions.pt")

    directions = {
        layer_idx: torch.mean(
            torch.stack([n * direction[pos, layer_idx]
                         for _, (n, direction) in category_directions.items()], dim=0),
            dim=0,
        )
        for layer_idx in range(1, num_layers)
    }

    torch.save(directions, refusal_directions_path)

    return directions


def load_directions(cfg, model_base, harmful_train, harmless_train,
                    batch_size=128, use_existing=True):
    dir_path          = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(dir_path, exist_ok=True)
    cat_means_path    = os.path.join(dir_path, "cat_means.pt")
    harmless_ref_path = os.path.join(dir_path, "harmless_reference.pt")
    if use_existing and os.path.exists(cat_means_path) and os.path.exists(harmless_ref_path):
        return torch.load(cat_means_path), torch.load(harmless_ref_path)
    cat_data, harmless_mean = generate_category_directions(
        model_base, harmful_train, harmless_train,
        artifact_dir=dir_path, batch_size=batch_size,
    )

    torch.save(harmless_mean, harmless_ref_path)
    return cat_data, harmless_mean


def make_embedding_hook(direction: torch.Tensor, mask_indices=None):
    """
    Output hook on embed_tokens: projects out the refusal direction
    from token embeddings as they enter the residual stream.

    If mask_indices is provided, only coordinates in the mask are projected
    (sparse orthogonalisation); otherwise the full dense projection is applied.
    """
    def hook_fn(module, input, output):
        dtype, device = output.dtype, output.device
        d      = direction.to(device=device, dtype=dtype)
        norm_d = d / (d.norm() + 1e-8)
        proj_out = (output @ norm_d).unsqueeze(-1) * norm_d   # [..., d_model]

        if mask_indices is not None:
            mask = torch.zeros(d.shape[0], dtype=torch.bool, device=device)
            mask[mask_indices] = True
            proj_out[..., ~mask] = 0.0                        # zero out non-mask coords

        return output - proj_out
    return hook_fn


# ── Hook ───────────────────────────────────────────────────────────────────────

def run_component_test(
    cfg,
    model_base,
    minimal_set,
    directions,
    anchor_layer: int,
    harmful_prompts,
    override_tag = None,
    ablation = False,
    batch_size: int = 128,
    use_existing: bool = False,
):
    
    hooks = []
    tags = []
    abbrev_tag = ""
    for name, module, hook in minimal_set:
        hooks.append((module, hook))
        abbrev_tag += name.replace("attn", "a").replace("mlp", "m").replace("embed", "e").replace("_","")
    if ablation:
        abbrev_tag += f"ablate_{ablation}"
    if override_tag:
        abbrev_tag = override_tag

    tag = f"CompTest_L{anchor_layer}_{abbrev_tag}"

    generate_and_save_completions_for_dataset(
        cfg,
        model_base,
        [],   # snapshot of current hook set
        hooks.copy(),
        tag,
        "harmful",
        batch_size=batch_size,
        dataset=harmful_prompts,
        save_path="completions/pruning" if (ablation) else "completions/component_test",
        use_existing=use_existing,
    )
    tags.append((tag, "completions/pruning" if (ablation) else "completions/component_test"))

    return tags

def compute_component_cos_sim(
    cfg,
    model_base,
    minimal_set,
    data,
    refusal_direction,
    prompt_type = "harmful",
    use_existing=False,
    batch_size=32,
    save_suffix="",
):
    device     = refusal_direction.device
    save_path  = os.path.join(cfg.artifact_path(), f"component_cossim{save_suffix}_{prompt_type}.json")
    norms_path = os.path.join(cfg.artifact_path(), f"component_mean_output_norms{save_suffix}_{prompt_type}.pt")

    if os.path.exists(save_path) and not use_existing:
        with open(save_path) as f:
            return json.load(f)

    # Ensure refusal direction is a unit vector in float32
    norm_d = (refusal_direction / (refusal_direction.norm() + 1e-8)).detach().cpu().float()
    results = {}
    mean_output_norms = {}

    for component_name, component_module, _ in minimal_set:
        print(f"Capturing: {component_name}...")
        
        # Local storage for this component's batch results
        last_token_acts = []

        def capture_hook(module, input, output):
            # Extract the tensor from potential tuples (hidden_states, [attentions/cache])
            out = output[0] if isinstance(output, tuple) else output
            
            last_act = out[:, -1, :].detach().cpu().float() 
            last_token_acts.append(last_act)

        hook = component_module.register_forward_hook(capture_hook)
        
        try:
            for start in range(0, len(data), batch_size):
                batch = data[start : start + batch_size]
                batch_prompts = [d["instruction"] if not isinstance(d, str) else d for d in batch]
                
                # Use left-padding for last-token indexing consistency
                model_base.tokenizer.padding_side = "left"
                inputs = model_base.tokenizer(
                    batch_prompts, 
                    return_tensors="pt",
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(device)
                
                with torch.no_grad():
                    model_base.model(**inputs)
        finally:
            hook.remove()

        # Combine all batches for this component -> [N_prompts, d]
        all_last_tokens = torch.cat(last_token_acts, dim=0)

        # Calculate L2 norm for each prompt (dim -1)
        # keepdim=True allows for easy division
        token_norms = all_last_tokens.norm(dim=-1, keepdim=True) + 1e-8
        unit_last_tokens = all_last_tokens / token_norms
        
        # Projection (now Cosine Similarity): [N, d] @ [d, 1] -> [N]
        projections = (unit_last_tokens @ norm_d).numpy()

        # Calculate Percentiles
        results[component_name] = {
            "q5":             float(np.percentile(projections, 5)),
            "q25":            float(np.percentile(projections, 25)),
            "median":         float(np.median(projections)),
            "q75":            float(np.percentile(projections, 75)),
            "q95":            float(np.percentile(projections, 95)),
            "mean":           float(projections.mean()),
            "component_type": "attention" if "attn" in component_name.lower() else "mlp",
        }

        # Save Mean Directional Write (for the scatter plots)
        mean_vec = all_last_tokens.mean(dim=0)
        mean_output_norms[component_name] = mean_vec / (mean_vec.norm() + 1e-8)

        print(f"  Done {component_name}: Med={results[component_name]['median']:.4f}")

    # Final Save
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    torch.save(mean_output_norms, norms_path)
    print(f"Artifacts saved to {cfg.artifact_path()}")

    return results


# ── Evaluation ─────────────────────────────────────────────────────────────────

import gc
import torch
import ctypes

def unload_model(model_base):
    if model_base is None:
        return

    # 1. Break internal references
    if hasattr(model_base, 'model') and model_base.model is not None:
        # Move to CPU first to free VRAM immediately
        model_base.model.to('cpu')
        # Manually clear the weight data if possible
        for param in model_base.model.parameters():
            param.data = torch.empty(0)
        model_base.model = None
        
    if hasattr(model_base, 'tokenizer'):
        model_base.tokenizer = None
        
    # 2. Force Garbage Collection
    gc.collect()
    
    # 3. Clear CUDA Cache & Synchronize
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 4. Release C-level memory back to OS
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass

from contextlib import contextmanager

@contextmanager
def active_hooks(hooks_list):
    try:
        yield hooks_list
    finally:
        for hook in hooks_list:
            hook.remove()

def evaluate_tags(cfg, tags_and_paths, batch_size, use_existing):
    print("\nEvaluating completions...")
    for tag, save_path in tags_and_paths:
        print(f"  {tag}")
        evaluate_completions_and_save_results_for_dataset(
            cfg, tag, "harmful",
            eval_methodologies = cfg.jailbreak_eval_methodologies,
            save_path=save_path,
            use_existing=use_existing,
            batch_size=batch_size,
        )
    unload_llamaguard_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_results(cfg, tags_and_paths):
    print("\n" + "=" * 60)
    print("COMPONENT SWEEP RESULTS")
    print("=" * 60)
    for tag, save_path in tags_and_paths:
        path = os.path.join(cfg.artifact_path(), save_path,
                            f"harmful_{tag}_evaluations.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            asr = data.get("llamaguard3_success_rate",
                           data.get("success_rate", "n/a"))
            print(f"  {tag:<55}  ASR={asr:.3f}" if isinstance(asr, float)
                  else f"  {tag:<55}  ASR={asr}")
        else:
            print(f"  {tag:<55}  [no evaluation file]")
    print("=" * 60)




# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Steer model via component (MLP/Attn) hooks walking back from anchor layer."
    )
    parser.add_argument("--model_path",     type=str, required=True)
    parser.add_argument("--anchor_layer",   type=int, required=True,
                        help="Layer to anchor the sweep at (start of walk-back).")
    parser.add_argument("--batch_size",     type=int, default=256)
    parser.add_argument("--n_test",         type=int, default=512)
    parser.add_argument("--top_energy",         type=float, default=0.95)
    parser.add_argument("--heuristic_threshold",         type=float, default=0.95)
    parser.add_argument("--sparse_threshold",         type=float, default=-1.)
    parser.add_argument("--use_existing",   action="store_true")

    args = parser.parse_args()

    cfg        = Config(model_alias=os.path.basename(args.model_path),
                        model_path=args.model_path)
    model_base = construct_model_base(args.model_path)

    # Data
    harmful_train  = load_dataset_split("mixed_harmful", "train", instructions_only=False)
    harmless_train = load_dataset_split("harmless",       "train", instructions_only=True)[:len(harmful_train)]
    harmful_val   = load_dataset_split("mixed_harmful", "val",  instructions_only=False)
    harmful_test   = load_dataset_split("mixed_harmful", "test",  instructions_only=False)[:args.n_test]

    # Directions
    category_directions, harmless_mean = load_directions(
        cfg, model_base, harmful_train, harmless_train,
        batch_size=args.batch_size, use_existing=args.use_existing,
    )
    first_dir  = next(iter(category_directions.values()))[1]
    num_layers = first_dir.shape[1]
    directions = build_directions(cfg, category_directions, num_layers)

    harmless_mean = harmless_mean[-1]
    target_direction = directions[args.anchor_layer]

    evaluation_tags = []


    deterministic_circuit = heuristic_elimination_search(
        cfg, model_base, directions, harmful_val, harmless_mean,
        heuristic_threshold = args.heuristic_threshold,
        anchor_layer=args.anchor_layer,
        batch_size=args.batch_size,
        use_existing=args.use_existing
    )

    print(f"Deterministic circuit: {len(deterministic_circuit)} components - {[c[0] for c in deterministic_circuit]}")

    # Clean up original model and extract just names to save memory
    deterministic_names = [c[0] for c in deterministic_circuit]
    del deterministic_circuit, model_base
    gc.collect()

    # Iteratively prune circuits
    print("\nPruning deterministic circuit...")
    model_base = construct_model_base(cfg.model_path)
    deterministic_names_rebuilt = rebuild_module_set(deterministic_names, model_base, directions, args.anchor_layer)
    deterministic_pruned = iterative_pruning(
        cfg, model_base, deterministic_names_rebuilt, directions, harmful_val,
        anchor_layer=args.anchor_layer,
        threshold=0.05,
        save_suffix="_determinstic",
        use_existing=args.use_existing
    )

    print(f"Final deterministic circuit: {len(deterministic_pruned)} components - {[c[0] for c in deterministic_pruned]}")

    # Extract names for analysis to avoid carrying module references
    deterministic_final_names = [c[0] for c in deterministic_pruned]
    del deterministic_pruned, model_base
    gc.collect()

    # Analyze both final circuits
    circuits_to_analyze = [
        ("deterministic", deterministic_final_names),
    ]

    for circuit_name, circuit_names in circuits_to_analyze:
        print(f"\n{'='*50}")
        print(f"Analyzing {circuit_name} circuit...")
        print(f"{'='*50}")
        
        # Fresh model for each analysis
        model_base = construct_model_base(cfg.model_path)
        circuit = rebuild_module_set(circuit_names.copy(), model_base, directions, args.anchor_layer)
        
        # Test performance
        print(f"Testing {circuit_name} circuit performance...")
        evaluation_tags = []
        tag = run_component_test(
            cfg, model_base, circuit, directions,
            override_tag = f"_{circuit_name}",
            anchor_layer=args.anchor_layer,
            harmful_prompts=harmful_test,
            batch_size=args.batch_size,
            use_existing=args.use_existing,
        )

        evaluation_tags += tag 

        complement_circuit = rebuild_module_set(circuit_names.copy(), model_base, directions, args.anchor_layer, complement = True)
        
        # Test performance
        print(f"Testing {circuit_name} complement circuit performance...")
        tag = run_component_test(
            cfg, model_base, complement_circuit, directions,
            override_tag = f"_{circuit_name}_complement",
            anchor_layer=args.anchor_layer,
            harmful_prompts=harmful_test,
            batch_size=args.batch_size,
            use_existing=args.use_existing,
        )

        evaluation_tags += tag 

        sparse_threshold = args.sparse_threshold
        if sparse_threshold > 0.0:

            target_direction = directions[args.anchor_layer]
            sparse_mask = get_norm_explained_mask(directions[args.anchor_layer], energy_threshold=sparse_threshold)

            vals = target_direction.detach().cpu().numpy().flatten()


        

            size = len(sparse_mask)
            # Rebuild with sparse hooks derived from the anchor-layer direction
            sparse_circuit = rebuild_module_set(
                circuit_names.copy(),       # copy so nullification doesn't destroy the list
                model_base,
                directions,
                args.anchor_layer,
                energy_threshold=sparse_threshold,
            )

            sparse_tag = run_component_test(
                cfg,
                model_base,
                sparse_circuit,
                directions,
                anchor_layer=args.anchor_layer,
                harmful_prompts=harmful_test,
                override_tag=f"_{circuit_name}_sparse_e{int(sparse_threshold*100)}_{size}coords",
                batch_size=args.batch_size,
                use_existing=args.use_existing,
            )

            evaluation_tags += sparse_tag 

        
        # SVD analysis
        print(f"Running SVD analysis on {circuit_name} circuit...")
        svd_results = analyze_minimal_set_svd(
            circuit, model_base, directions, args.anchor_layer, cfg, 
            save_suffix=f"_{circuit_name}",
            use_existing=args.use_existing
        )
        
        # Sparse overlap analysis
        print(f"Computing sparse overlap for {circuit_name} circuit...")
        results = compute_component_cos_sim(
            cfg,
            model_base,
            circuit,
            harmful_train,
            directions[args.anchor_layer],
            prompt_type = "harmful",
            save_suffix=f"_{circuit_name}",
            use_existing=args.use_existing
        )

        results = compute_component_cos_sim(
            cfg,
            model_base,
            circuit,
            harmless_train,
            directions[args.anchor_layer],
            prompt_type = "harmless",
            save_suffix=f"_{circuit_name}",
            use_existing=args.use_existing
        )
        
        unload_model(model_base)
        gc.collect()
        torch.cuda.empty_cache()
        
        # Evaluate performance
        evaluate_tags(cfg, evaluation_tags, batch_size=args.batch_size,
                    use_existing=args.use_existing)
        
        # Print results
        print(f"\n{circuit_name.title()} circuit results:")
        print_results(cfg, evaluation_tags)

    print(f"\n{'='*50}")
    print("Analysis complete!")
    print(f"{'='*50}")

    # Cleanup
    model_base.model     = None
    model_base.tokenizer = None
    del model_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()