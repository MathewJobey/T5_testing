import torch
print(torch.cuda.is_available())  # Should return True if GPU is detected
print(torch.cuda.get_device_name(0))  # Should show "RTX 3050"

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "models/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_path,legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")
