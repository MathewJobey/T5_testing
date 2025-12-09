from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os

# --------------------------------------------------
# 1. Setup & Load Model (TARGETING CHECKPOINT-565)
# --------------------------------------------------
# We point specifically to the checkpoint folder you mentioned
model_path = "./flan_t5_logs_rawonly_new/checkpoint-500"

print(f"Loading model from {model_path}...")

# Check if path exists to avoid confusion
if not os.path.exists(model_path):
    print(f"❌ Error: The folder '{model_path}' does not exist.")
    print("   Please check that you are running this script from the correct directory.")
    exit()

try:
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}")

# --------------------------------------------------
# 2. Prediction Function
# --------------------------------------------------
def generate_summary(log_text):
    inputs = tokenizer(
        log_text, 
        return_tensors="pt", 
        max_length=256, 
        truncation=True
    ).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

# --------------------------------------------------
# 3. Interactive Loop
# --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print(f"  LOG INTERPRETER (Checkpoint 500)")
    print("  Type 'exit' to quit")
    print("="*50)

    while True:
        raw_log = input("\n📝 Paste Raw Log:\n> ")
        
        if raw_log.lower() in ['exit', 'quit']:
            break
        
        if not raw_log.strip():
            continue

        result = generate_summary(raw_log)
        
        print(f"\n🤖 Interpretation:\n{result}")
        print("-" * 50)