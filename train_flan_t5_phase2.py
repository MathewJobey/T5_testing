from transformers import (
    TrainingArguments, 
    Trainer, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    DataCollatorForSeq2Seq  # <--- IMPORTED THIS
)
from datasets import load_dataset

# --------------------------------------------------
# Load dataset (RAW LOGS ONLY)
# --------------------------------------------------
# Ensure your CSV has headers: "input_text" and "target_text"
dataset = load_dataset("csv", data_files="tuning_dataset2.csv")

# dataset["train"] is a Dataset. train_test_split returns a DatasetDict with 'train' and 'test'
dataset = dataset["train"].train_test_split(test_size=0.1)

# --------------------------------------------------
# Load model + tokenizer
# --------------------------------------------------
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# --------------------------------------------------
# Preprocess function
# --------------------------------------------------
def preprocess(batch):
    # Tokenize inputs
    model_inputs = tokenizer(
        batch["input_text"],
        max_length=256,
        truncation=True,
    )

    # Tokenize targets (labels)
    # UPDATED: 'as_target_tokenizer' is deprecated. Use 'text_target' argument instead.
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# --------------------------------------------------
# Data Collator (REQUIRED for T5)
# --------------------------------------------------
# This handles dynamic padding to the longest sequence in the batch
# and correctly masks pad tokens in the labels with -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
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

    # Note: 'eval_strategy' requires Transformers >= 4.41.0
    # Use 'evaluation_strategy' if you are on an older version.
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,

    save_strategy="steps",
    save_steps=50,

    load_best_model_at_end=True,
    save_total_limit=2,
    gradient_accumulation_steps=2,

    fp16=False,
    remove_unused_columns=False,
    report_to="tensorboard",

    resume_from_checkpoint=None,
    logging_dir="./logs_rawonly",
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,  # <--- ADDED THIS
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