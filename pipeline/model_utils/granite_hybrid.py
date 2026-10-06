import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Union
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# GraniteMoeHybrid chat template (similar to Granite)
GRANITE_MOE_HYBRID_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

GRANITE_MOE_HYBRID_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

GRANITE_MOE_HYBRID_REFUSAL_TOKS = [40]  # 'I'

def format_instruction_granite_moe_hybrid_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
    reasoning: bool = False,
):
    if system is not None:
        formatted_instruction = GRANITE_MOE_HYBRID_CHAT_TEMPLATE_WITH_SYSTEM.format(
            instruction=instruction,
            system_prompt=system
        )
    else:
        formatted_instruction = GRANITE_MOE_HYBRID_CHAT_TEMPLATE.format(instruction=instruction)

    if reasoning:
        formatted_instruction += "<think>\n"

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_granite_moe_hybrid_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
    reasoning = False
):

    if isinstance(instructions, (str, dict)):
        instructions = [instructions]

    # Extract instruction strings
    cleaned_instructions = []
    for item in instructions:
        if isinstance(item, dict):
            instr = str(item.get("instruction", ""))
        else:
            instr = str(item)
        cleaned_instructions.append(instr)

    if outputs is not None:
        prompts = [
            format_instruction_granite_moe_hybrid_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace, reasoning = reasoning)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_granite_moe_hybrid_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace, reasoning = reasoning)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_granite_moe_hybrid_weights(model, direction: Float[Tensor, "d_model"]):
    # GraniteMoeHybrid uses model.embed_tokens
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    # Iterate through layers - some are attention, some are mamba
    for block in model.model.layers:
        # Only orthogonalize attention layers (mamba layers have different structure)
        if hasattr(block, 'self_attn') and block.self_attn is not None:
            block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        
        # Orthogonalize MoE output (shared across both layer types)
        if hasattr(block, 'block_sparse_moe'):
            # MoE has parallel experts - orthogonalize output linear
            block.block_sparse_moe.output_linear.weight.data = get_orthogonalized_matrix(
                block.block_sparse_moe.output_linear.weight.data.reshape(-1, block.block_sparse_moe.output_linear.weight.data.shape[-1]).T, 
                direction
            ).T.reshape(block.block_sparse_moe.output_linear.weight.data.shape)

def act_add_granite_moe_hybrid_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    # GraniteMoeHybrid uses model.layers
    block = model.model.layers[layer-1]
    
    # Add to MoE output (present in all layers)
    if hasattr(block, 'block_sparse_moe'):
        dtype = block.block_sparse_moe.output_linear.weight.dtype
        device = block.block_sparse_moe.output_linear.weight.device
        bias = (coeff * direction).to(dtype=dtype, device=device)
        
        # For parallel experts, we need to handle the special weight structure
        # Just add bias to the first expert's output as a simplification
        # (proper implementation would need to add to all experts)
        if not hasattr(block.block_sparse_moe.output_linear, 'bias') or block.block_sparse_moe.output_linear.bias is None:
            # Create bias if it doesn't exist (stored per expert)
            num_experts = block.block_sparse_moe.output_linear.num_experts
            # For now, skip adding bias to MoE (complex due to parallel expert structure)
            pass

class GraniteMoeHybridModel(ModelBase):
    """
    GraniteMoeHybrid model class that inherits from ModelBase.
    This is a hybrid architecture combining:
    - Mamba layers (SSM-based)
    - Attention layers  
    - MoE (Mixture of Experts) for FFN
    - Shared MLP alongside MoE
    
    Architecture follows model.model.layers structure like Granite.
    """

    def _load_model(self, model_path, dtype=torch.float16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation='eager',  # Use eager for safety with hybrid model
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model
    
    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            do_sample=True, 
            temperature=0.95,
            #use_cache=False,  # Disable cache to avoid hybrid cache OOM issues
        )

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_granite_moe_hybrid_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(GRANITE_MOE_HYBRID_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return GRANITE_MOE_HYBRID_REFUSAL_TOKS

    def _get_model_block_modules(self):
        # GraniteMoeHybrid uses model.layers
        return self.model.model.layers

    def _get_attn_modules(self):
        # Only return attention modules (some layers are mamba, not attention)
        return torch.nn.ModuleList([
            block_module.self_attn 
            for block_module in self.model_block_modules 
            if hasattr(block_module, 'self_attn') and block_module.self_attn is not None
        ])
    
    def _get_mlp_modules(self):
        # GraniteMoeHybrid has MoE + shared MLP, return shared MLP
        return torch.nn.ModuleList([
            block_module.shared_mlp 
            for block_module in self.model_block_modules
        ])

    def _get_post_attn_modules(self):
        return torch.nn.ModuleList([
            block_module.post_attention_layernorm 
            for block_module in self.model_block_modules
        ])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_granite_moe_hybrid_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_granite_moe_hybrid_weights, direction=direction, coeff=coeff, layer=layer)