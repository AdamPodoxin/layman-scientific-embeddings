import argparse
from pathlib import Path
import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
    )
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from peft import LoraConfig, TaskType
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


def parse_args():
    p = argparse.ArgumentParser("Finetune vanilla model")

    p.add_argument(
        "--model",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="Base embedding model to finetune",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="scibert",
        help="Base model name to save to output",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/scidocs/"),
        help="Output path to save model to",
    )
    p.add_argument(
        "--peft",
        action="store_true",
        help="Should use PEFT to reduce model size and training time?",
    )

    return p.parse_args()


def main():
    args = parse_args()

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

    model_kwargs = {}

    if args.peft:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config

    model = SentenceTransformer(args.model, model_kwargs=model_kwargs)

    if args.peft:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules="all-linear"
        )
        model.add_adapter(peft_config)

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH_SIZE)

    args = SentenceTransformerTrainingArguments(
        output_dir=Path(args.output) / f"vanilla-{args.model_name}",

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_sampler=BatchSamplers.NO_DUPLICATES,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        save_only_model=True,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        fp16=False,
        bf16=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=args,
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()