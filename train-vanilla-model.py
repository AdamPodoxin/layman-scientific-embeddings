from pathlib import Path
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
        losses, 
    )
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset, load_from_disk, concatenate_datasets


BASE_EMBEDDING_MODEL = "allenai/scibert_scivocab_uncased"

DATA_PATH = Path("data")

ABSTRACT_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "abstract-jargon"
ABSTRACT_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "abstract-layman"

TITLE_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "title-jargon"
TITLE_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "title-layman"

JARGON_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-jargon"
LAYMAN_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "layman-layman"
JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-layman"

MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-finetuned"
JARGON_LAYMAN_FINETUNED_MODEL_PATH = MODELS_PATH / "jargon-layman-finetuned"

# All combinations of jargon-jargon, layman-layman, and jargon-layman,
# as well as abstract-jargon, abstract-layman, title-jargon, and title-layman. 
NUM_PAIRS_PER_ABSTRACT = 15 * 14 * 3 + 15 * 4
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

PROP_PAIRS_TO_TAKE = 0.25


def main():
    abstract_jargon_pairs_dataset: Dataset = load_from_disk(str(ABSTRACT_JARGON_PAIRS_PATH))["train"]
    abstract_layman_pairs_dataset: Dataset = load_from_disk(str(ABSTRACT_LAYMAN_PAIRS_PATH))["train"]
    
    title_jargon_pairs_dataset: Dataset = load_from_disk(str(TITLE_JARGON_PAIRS_PATH))["train"]
    title_layman_pairs_dataset: Dataset = load_from_disk(str(TITLE_LAYMAN_PAIRS_PATH))["train"]

    jargon_jargon_pairs_dataset: Dataset = load_from_disk(str(JARGON_JARGON_PAIRS_PATH))["train"]
    layman_layman_pairs_dataset: Dataset = load_from_disk(str(LAYMAN_LAYMAN_PAIRS_PATH))["train"]
    jargon_layman_pairs_dataset: Dataset = load_from_disk(str(JARGON_LAYMAN_PAIRS_PATH))["train"]

    keyword_keyword_pairs_dataset = concatenate_datasets([
        jargon_jargon_pairs_dataset,
        layman_layman_pairs_dataset,
        jargon_layman_pairs_dataset,
    ])

    abstract_keyword_pairs_dataset = concatenate_datasets([
        abstract_jargon_pairs_dataset,
        abstract_layman_pairs_dataset,
    ])

    title_keyword_pairs_dataset = concatenate_datasets([
        title_jargon_pairs_dataset,
        title_layman_pairs_dataset,
    ])

    full_dataset = concatenate_datasets([
        keyword_keyword_pairs_dataset,
        abstract_keyword_pairs_dataset,
        title_keyword_pairs_dataset,
    ])

    # For efficiency, taking a subset of the entire pairs dataset
    train_dataset = full_dataset \
                                .shuffle() \
                                .take(int(PROP_PAIRS_TO_TAKE * full_dataset.shape[0]))

    vanilla_finetuned_model = SentenceTransformer(BASE_EMBEDDING_MODEL)

    loss = losses.CachedMultipleNegativesRankingLoss(vanilla_finetuned_model, mini_batch_size=MINI_BATCH_SIZE)

    args = SentenceTransformerTrainingArguments(
        output_dir=VANILLA_FINETUNED_MODEL_PATH,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = SentenceTransformerTrainer(
        model=vanilla_finetuned_model,
        train_dataset=train_dataset,
        loss=loss,
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    main()