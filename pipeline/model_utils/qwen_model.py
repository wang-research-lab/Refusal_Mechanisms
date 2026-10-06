# https://github.com/andyrdt/refusal_direction/blob/main/pipeline/model_utils/qwen_model.py
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Union
from torch import Tensor
from jaxtyping import Int, Float
import pdb


from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
  
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    # Ensure `instructions` is a list
    if isinstance(instructions, (str, dict)):
        instructions = [instructions]

    # Extract the instruction string from each element
    cleaned_instructions = []
    for item in instructions:
        if isinstance(item, dict):
            instr = str(item.get("instruction", ""))
        else:
            instr = str(item)
        cleaned_instructions.append(instr)

    # Construct prompt list
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(cleaned_instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in cleaned_instructions
        ]

    # Tokenize the prompts
    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T


def _is_moe_block(mlp_module) -> bool:
    """True if this layer's MLP is a sparse MoE block (Qwen3MoeSparseMoeBlock)
    rather than a standard dense MLP."""
    return hasattr(mlp_module, "experts")


def orthogonalize_qwen3moe_weights(model, direction: Float[Tensor, "d_model"], layer: int):
    """
    Orthogonalise every residual-stream-writing weight for layers 0..(layer-1)
    plus the embedding table, handling both dense and MoE MLP layers.

    Dense layers  — standard Linear down_proj, orthogonalise weight directly.
    MoE layers    — Qwen3MoeExperts.down_proj is a 3D parameter of shape
                    (num_experts, hidden_dim, intermediate_dim).
                    Each expert slice is (hidden_dim, intermediate_dim); the
                    d_model axis is dim-0, so no transpose is needed before
                    passing to get_orthogonalized_matrix (which expects rows
                    to be the d_model vectors), unlike the Linear case where
                    we transpose to bring d_model to the row axis then back.
    """
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for block in model.model.layers[:layer]:
        # Attention o_proj — always a standard Linear
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T

        # MLP down_proj — branch on dense vs MoE
        if _is_moe_block(block.mlp):
            # experts.down_proj: (num_experts, hidden_dim, intermediate_dim)
            # row axis (dim-1) is already hidden_dim == d_model
            experts = block.mlp.experts
            for expert_idx in range(experts.down_proj.data.shape[0]):
                # slice shape: (hidden_dim, intermediate_dim)
                experts.down_proj.data[expert_idx] = get_orthogonalized_matrix(
                    experts.down_proj.data[expert_idx].T, direction
                ).T
        else:
            # Standard Linear: weight shape (hidden_size, intermediate_size)
            # d_model is on the row axis already, but we follow the same
            # .T convention as o_proj for consistency with the rest of the
            # codebase (get_orthogonalized_matrix is applied to columns that
            # write into d_model, i.e. weight.T rows).
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.down_proj.weight.data.T, direction
            ).T


def _is_qwen3_moe(model_name_or_path: str) -> bool:
    """Detect Qwen3 MoE from the model path, e.g. Qwen/Qwen3-30B-A3B-Thinking-2507."""
    import re
    # Matches patterns like A3B, A2B, A22B anywhere in the path
    return bool(re.search(r"-A\d+B", model_name_or_path))


class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model_kwargs = {}
        if not _is_qwen3_moe(model_path):
            model_kwargs.update({"use_flash_attn": True})
        if dtype != "auto":
            model_kwargs.update({
                "bf16": dtype==torch.bfloat16,
                "fp16": dtype==torch.float16,
                "fp32": dtype==torch.float32,
            })

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=True, temperature = 0.6, top_p = 0.95)

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.pad_token_id = 151643

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        if _is_qwen3_moe(self.model_name_or_path):
            return self.model.model.layers
        elif "2" in self.model_name_or_path or "3" in self.model_name_or_path:
            return self.model.model.layers
        elif "Chat" in self.model_name_or_path:
            for i, layer in enumerate(self.model.transformer.h):
                self.model.transformer.h[i].self_attn = layer.attn
            return self.model.transformer.h
        else:
            raise AssertionError("model unknown - not 2, 2.5, 3, 3-MoE or Qwen 1")

    def _get_attn_modules(self):
        if _is_qwen3_moe(self.model_name_or_path):
            return torch.nn.ModuleList([layer.self_attn for layer in self.model.model.layers])
        elif "2" in self.model_name_or_path or "3" in self.model_name_or_path:
            return [layer.self_attn for layer in self.model.model.layers]
        elif "Chat" in self.model_name_or_path:
            for i, layer in enumerate(self.model.transformer.h):
                self.model.transformer.h[i].self_attn = layer.attn
            return self.model.transformer.h
        else:
            raise AssertionError("model unknown - not 2, 2.5, 3, 3-MoE or Qwen 1")
            
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_post_attn_modules(self):
        if "Chat" in self.model_name_or_path:
            return None
        return torch.nn.ModuleList([block_module.post_attention_layernorm for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"], layer: int):
        if _is_qwen3_moe(self.model_name_or_path):
            return functools.partial(orthogonalize_qwen3moe_weights, direction=direction, layer=layer)
        # Fall through to base class behaviour for non-MoE Qwen variants —
        # those call sites don't pass `layer` so we don't wrap them here.
        return super()._get_orthogonalization_mod_fn(direction)

    def _get_act_add_mod_fn(self, *args, **kwargs):
        return None