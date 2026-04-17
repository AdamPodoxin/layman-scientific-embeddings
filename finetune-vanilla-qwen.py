from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
    )
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict, load_from_disk, concatenate_datasets

DATA_PATH = Path("data")

ABSTRACT_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "abstract-jargon"
ABSTRACT_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "abstract-layman"

TITLE_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "title-jargon"
TITLE_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "title-layman"

JARGON_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-jargon"
LAYMAN_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "layman-layman"
JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-layman"

# All combinations of jargon-jargon, layman-layman, and jargon-layman,
# as well as abstract-jargon, abstract-layman, title-jargon, and title-layman. 
NUM_PAIRS_PER_ABSTRACT = 15 * 14 * 3 + 15 * 4
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 30

PROP_PAIRS_TO_TAKE = 0.25

MODEL_ID = "unsloth/Qwen3-Embedding-0.6B"


def main():
    abstract_jargon_pairs_dataset: DatasetDict = load_from_disk(str(ABSTRACT_JARGON_PAIRS_PATH))
    abstract_layman_pairs_dataset: DatasetDict = load_from_disk(str(ABSTRACT_LAYMAN_PAIRS_PATH))
    
    title_jargon_pairs_dataset: DatasetDict = load_from_disk(str(TITLE_JARGON_PAIRS_PATH))
    title_layman_pairs_dataset: DatasetDict = load_from_disk(str(TITLE_LAYMAN_PAIRS_PATH))

    jargon_jargon_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_JARGON_PAIRS_PATH))
    layman_layman_pairs_dataset: DatasetDict = load_from_disk(str(LAYMAN_LAYMAN_PAIRS_PATH))
    jargon_layman_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_LAYMAN_PAIRS_PATH))

    keyword_keyword_pairs_dataset_train = concatenate_datasets([
        jargon_jargon_pairs_dataset["train"],
        layman_layman_pairs_dataset["train"],
        jargon_layman_pairs_dataset["train"],
    ])

    abstract_keyword_pairs_dataset_train = concatenate_datasets([
        abstract_jargon_pairs_dataset["train"],
        abstract_layman_pairs_dataset["train"],
    ])

    title_keyword_pairs_dataset_train = concatenate_datasets([
        title_jargon_pairs_dataset["train"],
        title_layman_pairs_dataset["train"],
    ])

    full_dataset_train = concatenate_datasets([
        keyword_keyword_pairs_dataset_train,
        abstract_keyword_pairs_dataset_train,
        title_keyword_pairs_dataset_train,
    ])

    # For efficiency, taking a subset of the entire pairs dataset
    train_dataset = full_dataset_train \
                                .shuffle() \
                                .take(int(PROP_PAIRS_TO_TAKE * full_dataset_train.shape[0]))

    keyword_keyword_pairs_dataset_val = concatenate_datasets([
        jargon_jargon_pairs_dataset["val"],
        layman_layman_pairs_dataset["val"],
        jargon_layman_pairs_dataset["val"],
    ])

    abstract_keyword_pairs_dataset_val = concatenate_datasets([
        abstract_jargon_pairs_dataset["val"],
        abstract_layman_pairs_dataset["val"],
    ])

    title_keyword_pairs_dataset_val = concatenate_datasets([
        title_jargon_pairs_dataset["val"],
        title_layman_pairs_dataset["val"],
    ])

    full_dataset_val = concatenate_datasets([
        keyword_keyword_pairs_dataset_val,
        abstract_keyword_pairs_dataset_val,
        title_keyword_pairs_dataset_val,
    ])

    # For efficiency, taking a subset of the entire pairs dataset
    val_dataset = full_dataset_val \
                                .shuffle() \
                                .take(int(PROP_PAIRS_TO_TAKE * full_dataset_val.shape[0]))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )

    st_model = SentenceTransformer(
        MODEL_ID,
        model_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    )

    st_model._first_module().auto_model = prepare_model_for_kbit_training(
        st_model._first_module().auto_model,
        use_gradient_checkpointing=True,
    )

    st_model.add_adapter(lora_config)

    loss = losses.CachedMultipleNegativesRankingLoss(st_model, mini_batch_size=MINI_BATCH_SIZE)

    args = SentenceTransformerTrainingArguments(
        output_dir=Path("models/scidocs/vanilla-qwen"),

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_sampler=BatchSamplers.NO_DUPLICATES,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        save_only_model=True,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,

        fp16=False,
        bf16=True,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit", 
    )

    trainer = SentenceTransformerTrainer(
        model=st_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=args,
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()