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

JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-layman"

MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-scibert"
OUTPUT_MODEL_PATH = MODELS_PATH / "jargon-layman-scibert"

# All combinations of jargon-layman
NUM_PAIRS_PER_ABSTRACT = 15 * 14
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 2e-6
WEIGHT_DECAY = 2e-6
BATCH_SIZE = 30

PROP_PAIRS_TO_TAKE = 1.00


def main():
    jargon_layman_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_LAYMAN_PAIRS_PATH))
    
    train_pairs = jargon_layman_pairs_dataset["train"]
    val_pairs = jargon_layman_pairs_dataset["val"]

    train_dataset = train_pairs \
                    .shuffle() \
                    .take(int(PROP_PAIRS_TO_TAKE * train_pairs.shape[0]))
    
    val_dataset = val_pairs \
                    .shuffle() \
                    .take(int(PROP_PAIRS_TO_TAKE * val_pairs.shape[0]))

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