from transformers import (
    TrainingArguments, 
    Trainer, 
    T5ForConditionalGeneration, 
    T5Tokenizer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# --------------------------------------------------
# Load split datasets
# --------------------------------------------------
train_ds = load_dataset("csv", data_files="train.csv")["train"]
val_ds   = load_dataset("csv", data_files="val.csv")["train"]

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

    # Tokenize targets using text_target
    labels = tokenizer(
        text_target=batch["target_text"],
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_ds.map(
    preprocess,
    batched=True,
    remove_columns=train_ds.column_names
)

val_tokenized = val_ds.map(
    preprocess,
    batched=True,
    remove_columns=val_ds.column_names
)

# --------------------------------------------------
# Data Collator (dynamic padding + label masking)
# --------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# --------------------------------------------------
# Training arguments
# --------------------------------------------------
args = TrainingArguments(
    output_dir="./flan_t5_logs_rawonly_new",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=300,

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
    logging_dir="./logs_rawonly_new",
)

# --------------------------------------------------
# Trainer
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
)

# --------------------------------------------------
# Train
# --------------------------------------------------
trainer.train()

# --------------------------------------------------
# Save final model
# --------------------------------------------------
trainer.save_model("./flan_t5_logs_rawonly_new")
tokenizer.save_pretrained("./flan_t5_logs_rawonly_new")

print("Training complete! Model saved in flan_t5_logs_rawonly_new/")
