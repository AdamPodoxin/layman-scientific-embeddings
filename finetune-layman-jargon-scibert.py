from pathlib import Path
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
    )
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from datasets import DatasetDict, load_from_disk


DATA_PATH = Path("data")

LAYMAN_JARGON_PAIRS_PATH = DATA_PATH / "pairs" / "layman-jargon"

MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-scibert"
OUTPUT_MODEL_PATH = MODELS_PATH / "layman-jargon-scibert"

# All combinations of layman-jargon
NUM_PAIRS_PER_ABSTRACT = 15 * 14
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32


def main():
    layman_jargon_pairs_dataset: DatasetDict = load_from_disk(str(LAYMAN_JARGON_PAIRS_PATH))
    
    train_dataset = layman_jargon_pairs_dataset["train"]
    val_dataset = layman_jargon_pairs_dataset["val"]

    model = SentenceTransformer(str(VANILLA_FINETUNED_MODEL_PATH))

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