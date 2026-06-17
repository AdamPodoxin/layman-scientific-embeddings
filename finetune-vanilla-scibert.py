from pathlib import Path
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
    )
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from utils import load_pairs_dataset, clean_pairs_for_training


# All combinations of jargon-jargon, layman-layman, and layman-jargon,
# as well as jargon-abstract, layman-abstract, jargon-title, and layman-title. 
NUM_PAIRS_PER_ABSTRACT = 15 * 14 * 3 + 15 * 4
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 30

PROP_PAIRS_TO_TAKE = 0.25
PROP_PAIRS_TO_TAKE = 0.00025

MODEL_ID = "allenai/scibert_scivocab_uncased"

OUTPUT_MODEL_PATH = Path("models/vanilla-scibert")


def main():
    pairs_dataset = load_pairs_dataset()
    
    full_dataset_train = clean_pairs_for_training(pairs_dataset["train"])
    full_dataset_val = clean_pairs_for_training(pairs_dataset["val"])

    # For efficiency, taking a subset of the entire pairs dataset
    train_dataset = full_dataset_train.take(int(PROP_PAIRS_TO_TAKE * full_dataset_train.shape[0]))
    val_dataset = full_dataset_val.take(int(PROP_PAIRS_TO_TAKE * full_dataset_val.shape[0]))

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
