from pathlib import Path
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
    )
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from datasets import DatasetDict, load_from_disk, concatenate_datasets


DATA_PATH = Path("data")

JARGON_ABSTRACT_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-abstract"
LAYMAN_ABSTRACT_PAIRS_PATH = DATA_PATH / "pairs" / "layman-abstract"

JARGON_TITLE_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-title"
LAYMAN_TITLE_PAIRS_PATH = DATA_PATH / "pairs" / "layman-title"

JARGON_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-jargon"
LAYMAN_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "layman-layman"
LAYMAN_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "layman-jargon"

# All combinations of jargon-jargon, layman-layman, and layman-jargon,
# as well as jargon-abstract, layman-abstract, jargon-title, and layman-title. 
NUM_PAIRS_PER_ABSTRACT = 15 * 14 * 3 + 15 * 4
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 30

PROP_PAIRS_TO_TAKE = 0.25

MODEL_ID = "allenai/scibert_scivocab_uncased"

OUTPUT_MODEL_PATH = Path("models/vanilla-scibert")


def main():
    jargon_abstract_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_ABSTRACT_PAIRS_PATH))
    layman_abstract_pairs_dataset: DatasetDict = load_from_disk(str(LAYMAN_ABSTRACT_PAIRS_PATH))
    
    jargon_title_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_TITLE_PAIRS_PATH))
    layman_title_pairs_dataset: DatasetDict = load_from_disk(str(LAYMAN_TITLE_PAIRS_PATH))

    jargon_jargon_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_JARGON_PAIRS_PATH))
    layman_layman_pairs_dataset: DatasetDict = load_from_disk(str(LAYMAN_LAYMAN_PAIRS_PATH))
    layman_jargon_pairs_dataset: DatasetDict = load_from_disk(str(LAYMAN_JARGON_PAIRS_PATH))

    keyword_keyword_pairs_dataset_train = concatenate_datasets([
        jargon_jargon_pairs_dataset["train"],
        layman_layman_pairs_dataset["train"],
        layman_jargon_pairs_dataset["train"],
    ])

    abstract_keyword_pairs_dataset_train = concatenate_datasets([
        jargon_abstract_pairs_dataset["train"],
        layman_abstract_pairs_dataset["train"],
    ])

    title_keyword_pairs_dataset_train = concatenate_datasets([
        jargon_title_pairs_dataset["train"],
        layman_title_pairs_dataset["train"],
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
        layman_jargon_pairs_dataset["val"],
    ])

    abstract_keyword_pairs_dataset_val = concatenate_datasets([
        jargon_abstract_pairs_dataset["val"],
        layman_abstract_pairs_dataset["val"],
    ])

    title_keyword_pairs_dataset_val = concatenate_datasets([
        jargon_title_pairs_dataset["val"],
        layman_title_pairs_dataset["val"],
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


    model = SentenceTransformer(MODEL_ID)

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH_SIZE)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,

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