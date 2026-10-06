import torch
import functools

from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from typing import List, Union, Tuple
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Use actual refusal token(s) instead of placeholder
MISTRAL_REFUSAL_TOKS = [40]  # Update this as needed with real tokenizer values



def format_instruction_mistral_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    if not include_trailing_whitespace:
        instruction = instruction.strip()
    return f"<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{instruction}[/INST]"

def tokenize_instructions_mistral_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    # Normalize to list
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

    # Format prompts
    if outputs is not None:
        prompts = [
            format_instruction_mistral_chat(tokenizer, instruction, output, system, include_trailing_whitespace)
            for instruction, output in zip(cleaned_instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_mistral_chat(tokenizer, instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in cleaned_instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

def mistral_text_decoder(model):
    """Locate the text decoder backbone of a Mistral3ForConditionalGeneration,
    robust to the wrapper layout changing across transformers versions.

    transformers >=5.x:  model.model (Mistral3Model) -> .language_model -> .layers
    older layouts:        model.language_model -> .layers
    """
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    return getattr(model, "model", model)


def orthogonalize_mistral_weights(model, direction: Float[Tensor, "d_model"]):
    decoder = mistral_text_decoder(model)
    model.get_input_embeddings().weight.data = get_orthogonalized_matrix(
        model.get_input_embeddings().weight.data, direction
    )

    for block in decoder.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_mistral_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    layer_module = mistral_text_decoder(model).layers[layer]
    dtype = layer_module.mlp.down_proj.weight.dtype
    device = layer_module.mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    layer_module.mlp.down_proj.bias = torch.nn.Parameter(bias)

class MistralModel(ModelBase):

    # Mistral-Small-3.2 ships no tokenizer/processor of its own, so we borrow 3.1's
    # (see _load_tokenizer). save_ortho_weights reads this to write the image
    # processor (preprocessor_config.json) into baked checkpoints for vLLM.
    processor_source = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # bfloat16, NOT float16: Mistral-Small-24B is bf16-native and its activations
        # overflow fp16's ~65504 range -> NaNs -> gibberish (seen when the baked
        # checkpoint was served in fp16). bf16 matches the model's native dtype, so
        # this keeps both the baked weights and HF generation (sweep) sane.
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.requires_grad_(False)
        model.config = model.config.text_config
        return model

    def _load_tokenizer(self, model_path):
        #not sure why 3.2 doesn't include a tokenizer itself, hardcode 3.1 for now
        try:
            tokenizer = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503").tokenizer
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503", trust_remote_code=True)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=True, temperature=0.15, top_p=0.95)

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_mistral_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode("[/INST]", add_special_tokens=False)

    def _get_refusal_toks(self):
        return MISTRAL_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return mistral_text_decoder(self.model).layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self._get_model_block_modules()])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self._get_model_block_modules()])

    def _get_post_attn_modules(self):
        return torch.nn.ModuleList([block.post_attention_layernorm for block in self._get_model_block_modules()])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_mistral_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_mistral_weights, direction=direction, coeff=coeff, layer=layer)

    def compute_f_local(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute F_local = W_down W_down^T + W_O W_O^T for Mistral3.
        Accesses via model.language_model due to multimodal wrapper.
        Returns eigenvalues (descending), eigenvectors, F_local.
        """
        decoder = mistral_text_decoder(self.model)
        mlp_module = decoder.layers[layer_idx].mlp
        attn_module = decoder.layers[layer_idx].self_attn

        W_down = mlp_module.down_proj.weight.data.float()  # (hidden_size, intermediate_size)
        W_O = attn_module.o_proj.weight.data.float()       # (hidden_size, num_heads * head_dim)

        F_local = W_down @ W_down.T + W_O @ W_O.T

        eigenvalues, eigenvectors = torch.linalg.eigh(F_local)

        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        return eigenvalues, eigenvectors, F_local