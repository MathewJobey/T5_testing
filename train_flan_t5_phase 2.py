from transformers import TrainingArguments, Trainer, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

# --------------------------------------------------
# Load dataset (RAW LOGS ONLY)
# --------------------------------------------------
dataset = load_dataset("csv", data_files="tuning_dataset2.csv")
dataset = dataset["train"].train_test_split(test_size=0.1)

train_ds = dataset["train"]
eval_ds  = dataset["test"]

# --------------------------------------------------
# Load model + tokenizer
# --------------------------------------------------
model_path = "models/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# --------------------------------------------------
# Preprocess function
# --------------------------------------------------
def preprocess(batch):
    inputs = tokenizer(
        batch["input_text"],
        max_length=256,
        truncation=True,
        padding='max_length'
    )
    
    outputs = tokenizer(
        batch["target_text"],
        max_length=64,
        truncation=True,
        padding='max_length'
    )
    
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# --------------------------------------------------
# Training arguments
# --------------------------------------------------
args = TrainingArguments(
    output_dir="./flan_t5_logs_rawonly",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_rawonly",
    logging_steps=50,

    load_best_model_at_end=True,
    save_total_limit=2,
    gradient_accumulation_steps=2,
    fp16=True,                         # GPU mixed precision
    report_to="tensorboard",

    resume_from_checkpoint=None        # FIXED — avoids training crashes
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# --------------------------------------------------
# Train
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# Save final model
# --------------------------------------------------
trainer.save_model("./flan_t5_logs_rawonly")
tokenizer.save_pretrained("./flan_t5_logs_rawonly")

print("Training complete! Model saved in flan_t5_logs_rawonly/")
