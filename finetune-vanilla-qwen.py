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

from utils import load_pairs_dataset, clean_pairs_for_training


# All combinations of jargon-jargon, layman-layman, and layman-jargon,
# as well as jargon-abstract, layman-abstract, jargon-title, and layman-title. 
NUM_PAIRS_PER_ABSTRACT = (15 * 14 * 2) + (15 * 15) + (15 * 4)
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16

PROP_PAIRS_TO_TAKE = 0.25

MODEL_ID = "unsloth/Qwen3-Embedding-0.6B"

OUTPUT_MODEL_PATH = Path("models/vanilla-qwen")


def get_document_prompt(model: SentenceTransformer) -> str:
    for prompt_name in ("document", "passage", "corpus"):
        if prompt_name in model.prompts:
            return model.prompts[prompt_name]
    return ""


def main():
    pairs_dataset = load_pairs_dataset()
    full_dataset_train = clean_pairs_for_training(pairs_dataset["train"])
    full_dataset_val = clean_pairs_for_training(pairs_dataset["val"])

    # For efficiency, taking a subset of the entire pairs dataset
    train_dataset = full_dataset_train.take(int(PROP_PAIRS_TO_TAKE * full_dataset_train.shape[0]))
    val_dataset = full_dataset_val.take(int(PROP_PAIRS_TO_TAKE * full_dataset_val.shape[0]))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = SentenceTransformer(
        MODEL_ID,
        model_kwargs={
            "quantization_config": bnb_config,
            "device_map": "auto",
        }
    )

    lora_config = LoraConfig(
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

    model.add_adapter(lora_config)
    model.set_adapter("default")

    column_prompts = {
        "anchor": model.prompts.get("query", ""),
        "positive": get_document_prompt(model),
    }

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

        gradient_accumulation_steps=4,
        gradient_checkpointing=True,

        fp16=False,
        bf16=True,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",
        prompts=column_prompts,
        router_mapping={"anchor": "query", "positive": "document"},
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
