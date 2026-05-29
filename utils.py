from pathlib import Path
import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from datasets import Dataset, DatasetDict, load_from_disk
import math


QWEN_MODEL_ID = "unsloth/Qwen3-Embedding-0.6B"

PAIRS_PATH = Path("data") / "pairs"

KEYWORD_KEYWORD_PAIR_TYPES = frozenset({
    "jargon-jargon",
    "layman-layman",
    "layman-jargon",
})

DOC_KEYWORD_PAIR_TYPES = frozenset({
    "jargon-abstract",
    "layman-abstract",
    "jargon-title",
    "layman-title",
})


def load_pairs_dataset(path: Path | str = PAIRS_PATH) -> DatasetDict:
    return load_from_disk(str(path))


def filter_pairs_by_type(ds: Dataset, pair_types: set[str] | frozenset[str]) -> Dataset:
    return ds.filter(lambda example: example["pair_type"] in pair_types)


def clean_pairs_for_training(ds: Dataset, pair_types: set[str] | frozenset[str] | None = None) -> Dataset:
    if pair_types is not None:
        ds = filter_pairs_by_type(ds, pair_types)

    return ds.remove_columns(["pair_type", "doc_id"])


def load_qwen_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_qwen_base_4bit() -> SentenceTransformer:
    return SentenceTransformer(
        QWEN_MODEL_ID,
        model_kwargs={
            "quantization_config": load_qwen_bnb_config(),
            "device_map": "auto",
        },
    )


def load_finetuned_qwen(adapter: Path | str) -> SentenceTransformer:
    if isinstance(adapter, str):
        adapter = Path(adapter)

    model = load_qwen_base_4bit()
    first_module = model._first_module()
    first_module.auto_model = PeftModel.from_pretrained(
        first_module.auto_model,
        str(adapter),
        is_trainable=False,
    )
    return model


def round_down_to_2_decimals(x: float):
    return math.floor(x * 100) / 100
