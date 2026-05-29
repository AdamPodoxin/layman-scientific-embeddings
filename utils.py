from pathlib import Path
import argparse
import math

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from peft import LoraConfig, PeftModel, TaskType
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig


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

VANILLA_PAIR_TYPES = KEYWORD_KEYWORD_PAIR_TYPES | DOC_KEYWORD_PAIR_TYPES

# Excludes same-language keyword pairs that dilute cross-lingual retrieval signal.
VANILLA_PAIR_TYPES_FILTERED = (VANILLA_PAIR_TYPES - frozenset({
    "jargon-jargon",
    "layman-layman",
}))

DEFAULT_LORA_CONFIG = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
    use_qalora=True,
    use_rslora=True,
)


def load_pairs_dataset(path: Path | str = PAIRS_PATH) -> DatasetDict:
    return load_from_disk(str(path))


def filter_pairs_by_type(ds: Dataset, pair_types: set[str] | frozenset[str]) -> Dataset:
    return ds.filter(lambda example: example["pair_type"] in pair_types)


def clean_pairs_for_training(ds: Dataset, pair_types: set[str] | frozenset[str] | None = None) -> Dataset:
    if pair_types is not None:
        ds = filter_pairs_by_type(ds, pair_types)

    return ds.remove_columns(["pair_type", "doc_id"])


def load_test_pairs(path: Path | str = PAIRS_PATH) -> Dataset:
    return load_pairs_dataset(path)["test"]


def build_keyword_search_df(test_pairs: Dataset | None = None) -> pd.DataFrame:
    if test_pairs is None:
        test_pairs = load_test_pairs()

    layman_jargon = filter_pairs_by_type(test_pairs, {"layman-jargon"})
    return pd.DataFrame({
        "doc_id": layman_jargon["doc_id"],
        "layman": layman_jargon["anchor"],
        "jargon": layman_jargon["positive"],
    })


def build_layman_document_search_dfs(
    pair_type: str,
    test_pairs: Dataset | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_pairs is None:
        test_pairs = load_test_pairs()

    pairs = filter_pairs_by_type(test_pairs, {pair_type})
    query_df = pd.DataFrame({
        "doc_id": pairs["doc_id"],
        "layman": pairs["anchor"],
    })
    corpus_df = pd.DataFrame({
        "doc_id": pairs["doc_id"],
        "document": pairs["positive"],
    }).drop_duplicates(subset="doc_id").reset_index(drop=True)

    return query_df, corpus_df


def upsample_layman_jargon_pairs(ds: Dataset, weight: int) -> Dataset:
    if weight <= 1:
        return ds

    layman_jargon = ds.filter(lambda example: example["pair_type"] == "layman-jargon")
    other_pairs = ds.filter(lambda example: example["pair_type"] != "layman-jargon")
    return concatenate_datasets([layman_jargon] * weight + [other_pairs])


def prepare_vanilla_qwen_datasets(
    pairs_dataset: DatasetDict,
    prop_pairs_to_take: float,
    pair_types: set[str] | frozenset[str] = VANILLA_PAIR_TYPES_FILTERED,
    layman_jargon_weight: int = 2,
) -> tuple[Dataset, Dataset]:
    train_ds = filter_pairs_by_type(pairs_dataset["train"], pair_types)
    val_ds = filter_pairs_by_type(pairs_dataset["val"], pair_types)

    train_ds = upsample_layman_jargon_pairs(train_ds, layman_jargon_weight)

    train_size = int(prop_pairs_to_take * train_ds.shape[0])
    val_size = int(prop_pairs_to_take * val_ds.shape[0])

    return (
        clean_pairs_for_training(train_ds.take(train_size)),
        clean_pairs_for_training(val_ds.take(val_size)),
    )


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


def load_qwen_base_bf16(device: str = "cuda") -> SentenceTransformer:
    return SentenceTransformer(
        QWEN_MODEL_ID,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        },
        device=device,
    )


def load_qwen_for_training(use_4bit: bool = True) -> SentenceTransformer:
    if use_4bit:
        return load_qwen_base_4bit()
    return load_qwen_base_bf16()


def add_lora_adapter(model: SentenceTransformer) -> SentenceTransformer:
    model.add_adapter(DEFAULT_LORA_CONFIG)
    model.set_adapter("default")
    return model


def load_qwen_with_lora_adapter(
    adapter_path: Path | str,
    use_4bit: bool = True,
    is_trainable: bool = False,
) -> SentenceTransformer:
    model = load_qwen_for_training(use_4bit=use_4bit)
    first_module = model._first_module()
    first_module.auto_model = PeftModel.from_pretrained(
        first_module.auto_model,
        str(adapter_path),
        is_trainable=is_trainable,
    )
    return model


def is_lora_adapter_path(model_path: Path) -> bool:
    return (model_path / "adapter_config.json").exists()


def load_vanilla_qwen_for_second_stage(
    vanilla_path: Path | str,
    use_4bit: bool = True,
) -> SentenceTransformer:
    vanilla_path = Path(vanilla_path)

    if is_lora_adapter_path(vanilla_path):
        return load_qwen_with_lora_adapter(
            vanilla_path,
            use_4bit=use_4bit,
            is_trainable=True,
        )

    if use_4bit:
        raise ValueError(
            "Merged vanilla models require --bf16-lora for the layman-jargon stage. "
            "Use an adapter checkpoint or pass --bf16-lora."
        )

    model = SentenceTransformer(str(vanilla_path), device="cuda")
    add_lora_adapter(model)
    return model


def merge_lora_and_save(model: SentenceTransformer, output_path: Path | str) -> Path:
    output_path = Path(output_path)
    first_module = model._first_module()
    if isinstance(first_module.auto_model, PeftModel):
        first_module.auto_model = first_module.auto_model.merge_and_unload()
    model.save_pretrained(output_path)
    return output_path


def load_finetuned_qwen(model_path: Path | str) -> SentenceTransformer:
    if isinstance(model_path, str):
        model_path = Path(model_path)

    if is_lora_adapter_path(model_path):
        return load_qwen_with_lora_adapter(model_path, use_4bit=True, is_trainable=False)

    return SentenceTransformer(str(model_path), device="cuda")


def add_qwen_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bf16-lora",
        action="store_true",
        help="Train LoRA in bf16 without 4-bit quantization",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA weights into the base model before saving",
    )


def get_qwen_optimizer(use_4bit: bool) -> str:
    return "paged_adamw_8bit" if use_4bit else "adamw_torch"


def round_down_to_2_decimals(x: float):
    return math.floor(x * 100) / 100
