import unsloth

import argparse
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from utils import (
    add_qwen_training_args,
    clean_pairs_for_training,
    get_qwen_optimizer,
    load_pairs_dataset,
    load_vanilla_qwen_unsloth_for_second_stage,
    merge_lora_and_save,
)


MODEL_ID = "unsloth/Qwen3-Embedding-0.6B"

MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-qwen"
OUTPUT_MODEL_PATH = MODELS_PATH / "layman-jargon-qwen"

# All combinations of layman-jargon
NUM_PAIRS_PER_ABSTRACT = 15 * 15
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen LoRA on layman-jargon pairs")
    add_qwen_training_args(parser)
    parser.add_argument(
        "--vanilla-model-path",
        type=Path,
        default=VANILLA_FINETUNED_MODEL_PATH,
        help="Path to the vanilla fine-tuned LoRA adapter or merged model",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_MODEL_PATH,
        help="Directory to save the fine-tuned model",
    )
    return parser.parse_args()


def get_document_prompt(model: SentenceTransformer) -> str:
    for prompt_name in ("document", "passage", "corpus"):
        if prompt_name in model.prompts:
            return model.prompts[prompt_name]
    return ""


def count_trainable_parameters(model: SentenceTransformer) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    use_4bit = not args.bf16_lora

    pairs_dataset = load_pairs_dataset()
    train_dataset = clean_pairs_for_training(pairs_dataset["train"], {"layman-jargon"})
    val_dataset = clean_pairs_for_training(pairs_dataset["val"], {"layman-jargon"})

    model = load_vanilla_qwen_unsloth_for_second_stage(
        args.vanilla_model_path,
        use_4bit=use_4bit,
    )

    trainable_params = count_trainable_parameters(model)
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters after loading vanilla adapter. "
            "Check that is_trainable=True and the adapter path is valid."
        )
    print(f"Trainable parameters: {trainable_params:,}")

    column_prompts = {
        "anchor": model.prompts.get("query", ""),
        "positive": get_document_prompt(model),
    }

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH_SIZE)

    args_training = SentenceTransformerTrainingArguments(
        output_dir=args.output_path,

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
        optim=get_qwen_optimizer(use_4bit),
        prompts=column_prompts,
        router_mapping={"anchor": "query", "positive": "document"},
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=args_training,
    )

    trainer.train()

    if args.merge_lora:
        merge_lora_and_save(model, args.output_path)
    else:
        trainer.save_model()


if __name__ == "__main__":
    main()
