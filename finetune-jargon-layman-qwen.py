from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import BitsAndBytesConfig
from sentence_transformers import (
        SentenceTransformer, 
        SentenceTransformerTrainer, 
        SentenceTransformerTrainingArguments,
    )
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from peft import LoraConfig, TaskType
from datasets import DatasetDict, load_from_disk


DATA_PATH = Path("data")

JARGON_LAYMAN_PAIRS_PATH = DATA_PATH / "pairs" / "jargon-layman"

MODEL_ID = "unsloth/Qwen3-Embedding-0.6B"

MODELS_PATH = Path("models")
VANILLA_FINETUNED_MODEL_PATH = MODELS_PATH / "vanilla-qwen"
OUTPUT_MODEL_PATH = MODELS_PATH / "jargon-layman-qwen"

# All combinations of jargon-layman
NUM_PAIRS_PER_ABSTRACT = 15 * 14
NUM_ABSTRACTS_IN_BATCH = 10
MINI_BATCH_SIZE = NUM_PAIRS_PER_ABSTRACT * NUM_ABSTRACTS_IN_BATCH

LEARNING_RATE = 2e-6
WEIGHT_DECAY = 2e-6
BATCH_SIZE = 256


def main():
    jargon_layman_pairs_dataset: DatasetDict = load_from_disk(str(JARGON_LAYMAN_PAIRS_PATH))
    
    train_dataset = jargon_layman_pairs_dataset["train"]
    val_dataset = jargon_layman_pairs_dataset["val"]

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
    )

    model._first_module().auto_model.add_adapter(lora_config)

    adapter_state_dict = load_file(VANILLA_FINETUNED_MODEL_PATH / "adapter_model.safetensors")

    remapped = {
        k.replace("base_model.model.", ""): v 
        for k, v in adapter_state_dict.items()
    }

    model._first_module().auto_model.load_state_dict(remapped, strict=False)

    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH_SIZE)

    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_sampler=BatchSamplers.NO_DUPLICATES,

        eval_strategy="epoch",
        save_strategy="epoch",
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
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        args=args,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    main()
