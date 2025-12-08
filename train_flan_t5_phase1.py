from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration, T5Tokenizer

from datasets import load_dataset
import pandas as pd

# Load and optionally split dataset
dataset = load_dataset("csv", data_files="tuning_dataset.csv")
dataset=dataset["train"].train_test_split(test_size=0.1)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# --- Load local model and tokenizer ---
model_path = "models/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_path,legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def preprocess(batch):
    inputs = tokenizer(batch["input_text"], max_length=256, truncation=True,padding='max_length')
    outputs = tokenizer(batch["target_text"], max_length=128, truncation=True, padding='max_length')
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# --- Training arguments ---
args = TrainingArguments(
    output_dir="./flan_t5_logs_finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,                      # ✅ Added L2 regularization
    warmup_steps=500,                       # helps smooth the start of training
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    save_total_limit=2,                     # keep last 2 checkpoints only
    gradient_accumulation_steps=2,          # helpful for small GPU VRAM
    fp16=True,                              # mixed precision if GPU supports it
    resume_from_checkpoint=True,
    report_to="tensorboard"                       # log to TensorBoard
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()

trainer.save_model("./flan_t5_logs_finetuned")
tokenizer.save_pretrained("./flan_t5_logs_finetuned")
