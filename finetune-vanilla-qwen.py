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
    VANILLA_PAIR_TYPES,
    VANILLA_PAIR_TYPES_FILTERED,
    add_qwen_training_args,
    get_qwen_optimizer,
    load_pairs_dataset,
    load_qwen_unsloth_for_training,
    merge_lora_and_save,
    prepare_vanilla_qwen_datasets,
)


LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
# Tuned for 16GB VRAM (RTX 2000 Ada): bf16 LoRA + large physical batch.
BATCH_SIZE = 192
MINI_BATCH_SIZE = BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = 1

PROP_PAIRS_TO_TAKE = 1.00
LAYMAN_JARGON_WEIGHT = 2

OUTPUT_MODEL_PATH = Path("models/vanilla-qwen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA on mixed retrieval pairs")
    add_qwen_training_args(parser)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_MODEL_PATH,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--include-same-language-pairs",
        action="store_true",
        help="Include jargon-jargon and layman-layman pairs in vanilla training",
    )
    parser.add_argument(
        "--layman-jargon-weight",
        type=int,
        default=LAYMAN_JARGON_WEIGHT,
        help="Repeat layman-jargon pairs this many times to upweight them",
    )
    parser.add_argument(
        "--4bit-lora",
        dest="four_bit_lora",
        action="store_true",
        help="Use 4-bit quantization (saves VRAM but underutilizes a 16GB GPU)",
    )
    return parser.parse_args()


def get_document_prompt(model: SentenceTransformer) -> str:
    for prompt_name in ("document", "passage", "corpus"):
        if prompt_name in model.prompts:
            return model.prompts[prompt_name]
    return ""


def main():
    args = parse_args()
    # bf16 LoRA by default; 4-bit leaves ~90% of a 16GB GPU idle on this model.
    use_4bit = args.four_bit_lora and not args.bf16_lora
    pair_types = VANILLA_PAIR_TYPES if args.include_same_language_pairs else VANILLA_PAIR_TYPES_FILTERED

    pairs_dataset = load_pairs_dataset()
    train_dataset, val_dataset = prepare_vanilla_qwen_datasets(
        pairs_dataset,
        PROP_PAIRS_TO_TAKE,
        pair_types=pair_types,
        layman_jargon_weight=args.layman_jargon_weight,
    )

    model = load_qwen_unsloth_for_training(use_4bit=use_4bit)

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

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=False,

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
