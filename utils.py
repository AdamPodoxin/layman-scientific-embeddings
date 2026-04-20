from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from peft import LoraConfig


QWEN_MODEL_ID = "unsloth/Qwen3-Embedding-0.6B"


def load_finetuned_qwen(adapter: Path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = SentenceTransformer(
        QWEN_MODEL_ID,
        model_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    )

    lora_config = LoraConfig()

    model._first_module().auto_model.add_adapter(lora_config)

    adapter_state_dict = load_file(adapter / "adapter_model.safetensors")

    remapped = {
        k.replace("base_model.model.", ""): v 
        for k, v in adapter_state_dict.items()
    }

    model._first_module().auto_model.load_state_dict(remapped, strict=False)

    return model