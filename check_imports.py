from transformers import TrainingArguments
import inspect, transformers

print("Transformers version:", transformers.__version__)
print("Transformers location:", transformers.__file__)
print("TrainingArguments location:", inspect.getfile(TrainingArguments))
