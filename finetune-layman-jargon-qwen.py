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
from peft import PeftModel

from utils import load_pairs_dataset, clean_pairs_for_training


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


def get_document_prompt(model: SentenceTransformer) -> str:
    for prompt_name in ("document", "passage", "corpus"):
        if prompt_name in model.prompts:
            return model.prompts[prompt_name]
    return ""


def count_trainable_parameters(model: SentenceTransformer) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    pairs_dataset = load_pairs_dataset()
    train_dataset = clean_pairs_for_training(pairs_dataset["train"], {"layman-jargon"})
    val_dataset = clean_pairs_for_training(pairs_dataset["val"], {"layman-jargon"})

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

    first_module = model._first_module()
    first_module.auto_model = PeftModel.from_pretrained(
        first_module.auto_model,
        str(VANILLA_FINETUNED_MODEL_PATH),
        is_trainable=True,
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
