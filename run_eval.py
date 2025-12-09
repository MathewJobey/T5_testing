import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import torch

# ------------------------------------------------------------
# 1. Load your labeled dataset
# ------------------------------------------------------------
df = pd.read_csv("test.csv")   # use the test split now  # <--- your dataset
print("Loaded", len(df), "samples.")

# Ensure correct columns exist
if "input_text" not in df.columns or "target_text" not in df.columns:
    raise ValueError("CSV must contain: input_text, target_text")

# ------------------------------------------------------------
# 2. Load your fine-tuned model and enable CUDA
# ------------------------------------------------------------
model_path = "./flan_t5_logs_rawonly_new/checkpoint-500"

tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)
model.eval()

# ------------------------------------------------------------
# 3. Metric setup + GPU-enabled generation
# ------------------------------------------------------------
smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def generate_output(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

bleu_scores = []
rouge_scores = []
preds = []
targets = []

print("Running evaluation...\n")

# ------------------------------------------------------------
# 4. Run predictions + metrics
# ------------------------------------------------------------
for i in range(len(df)):
    raw_log = df.iloc[i]["input_text"]
    target = df.iloc[i]["target_text"]

    pred = generate_output(raw_log)

    preds.append(pred)
    targets.append(target)

    # BLEU
    bleu = sentence_bleu([target.split()], pred.split(), smoothing_function=smooth)
    bleu_scores.append(bleu)

    # ROUGE-L
    rougeL = rouge.score(target, pred)["rougeL"].fmeasure
    rouge_scores.append(rougeL)

# ------------------------------------------------------------
# 5. BERTScore (semantic similarity) — also runs on GPU
# ------------------------------------------------------------
P, R, F1 = bert_score.score(preds, targets, lang="en", verbose=True, device=str(device))
bert_f1 = float(F1.mean())

# ------------------------------------------------------------
# 6. Print summary
# ------------------------------------------------------------
print("\n===== FINAL EVALUATION =====")
print("Avg BLEU Score     :", sum(bleu_scores) / len(bleu_scores))
print("Avg ROUGE-L Score  :", sum(rouge_scores) / len(rouge_scores))
print("Avg BERTScore F1   :", bert_f1)

# ------------------------------------------------------------
# 7. Sample predictions
# ------------------------------------------------------------
print("\n===== SAMPLE OUTPUTS (first 5) =====")
for i in range(min(5, len(df))):
    print(f"\nSample #{i+1}")
    print("INPUT LOG :", df.iloc[i]['input_text'])
    print("TARGET    :", targets[i])
    print("PREDICTED :", preds[i])
