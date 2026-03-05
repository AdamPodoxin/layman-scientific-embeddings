from pathlib import Path
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
        losses, 
    )
from datasets import load_dataset


EMBEDDING_MODEL = "allenai/scibert_scivocab_uncased"

PAIRS_PATH = Path("data") / "all_pairs.parquet.gzip"
MODEL_PATH = Path("models") / "scideberta-full-jargon-layman-keywords"

# All combinations of jargon-jargon, layman-layman, and jargon-layman
NUM_PAIRS_PER_ABSTRACT = 15 * 14 * 3
NUM_ABSRACTS_IN_BATCH = 5
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSRACTS_IN_BATCH

LEARNING_RATE = 1e-5


def main():
    train_dataset = load_dataset("parquet", data_files=str(PAIRS_PATH))

    model = SentenceTransformer(EMBEDDING_MODEL)

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH_SIZE)

    trainer_args = SentenceTransformerTrainingArguments(
        output_dir=MODEL_PATH,
        learning_rate=LEARNING_RATE
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
        args=trainer_args,
    )
    trainer.train()

    model.save_pretrained(MODEL_PATH)


if __name__ == "__main__":
    main()