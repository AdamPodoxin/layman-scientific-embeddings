from pathlib import Path
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
        losses, 
    )
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset, load_dataset


BASE_EMBEDDING_MODEL = "allenai/scibert_scivocab_uncased"

DATA_PATH = Path("data")
ALL_PAIRS_PATH = DATA_PATH / "all_pairs.parquet.gzip"
JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "jargon-layman_pairs.parquet.gzip"

MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-finetuned"
JARGON_LAYMAN_FINETUNED_MODEL_PATH = MODELS_PATH / "jargon-layman-finetuned"

# All combinations of jargon-jargon, layman-layman, and jargon-layman
NUM_PAIRS_PER_ABSTRACT = 15 * 14 * 3
NUM_ABSRACTS_IN_BATCH = 5
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

JARGON_LAYMAN_PARAMS_SCALE = 0.5


def train(
        model: SentenceTransformer,
        output_dir: str,
        train_dataset: Dataset,
        learning_rate: float,
        weight_decay: float,
    ):

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH_SIZE)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        args=args,
    )

    return trainer.train()


def main():
    # Vanilla finetuned model
    all_pairs_train_dataset = load_dataset("parquet", data_files=str(ALL_PAIRS_PATH))

    vanilla_finetuned_model = SentenceTransformer(BASE_EMBEDDING_MODEL)

    train(
        model=vanilla_finetuned_model,
        output_dir=VANILLA_FINETUNED_MODEL_PATH,
        train_dataset=all_pairs_train_dataset,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    vanilla_finetuned_model.save_pretrained(VANILLA_FINETUNED_MODEL_PATH)


    # Jargon-Layman finetuned model
    jargon_layman_pairs_train_dataset = load_dataset("parquet", data_files=str(JARGON_LAYMAN_PAIRS_PATH))
    
    jargon_layman_finetuned_model = SentenceTransformer(str(VANILLA_FINETUNED_MODEL_PATH))

    train(
        model=jargon_layman_finetuned_model,
        output_dir=JARGON_LAYMAN_FINETUNED_MODEL_PATH,
        train_dataset=jargon_layman_pairs_train_dataset,
        learning_rate=LEARNING_RATE * JARGON_LAYMAN_PARAMS_SCALE,
        weight_decay=WEIGHT_DECAY * JARGON_LAYMAN_PARAMS_SCALE,
    )

    jargon_layman_finetuned_model.save_pretrained(JARGON_LAYMAN_FINETUNED_MODEL_PATH)


if __name__ == "__main__":
    main()