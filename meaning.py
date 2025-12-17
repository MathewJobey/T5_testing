import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# ==========================================
# 1. SETUP PHI-3
# ==========================================
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading {MODEL_ID}...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map=device, 
        torch_dtype="auto", 
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# 2. GENERATION FUNCTION
# ==========================================
def explain_log_phi3(template):
    # Clean text
    clean_text = template.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    
    messages = [
        {"role": "user", "content": f"You are a Linux Expert. Explain this log in one short sentence: '{clean_text}'"},
    ]
    
    try:
        output = pipe(
            messages, 
            max_new_tokens=60, 
            return_full_text=False, 
            temperature=0.1, 
            do_sample=False, 
        )
        return output[0]['generated_text'].strip()
    except Exception as e:
        return f"[Error] {e}"

# ==========================================
# 3. EXECUTION
# ==========================================
print("="*40)
filename = input("Enter the Excel filename (e.g., Linux_2k_clean_analysis.xlsx): ").strip()

if not filename or not os.path.exists(filename):
    print("File not found.")
    exit()

print(f"Reading {filename}...")
try:
    df_summary = pd.read_excel(filename, sheet_name="Template Summary", engine='openpyxl')
    df_logs = pd.read_excel(filename, sheet_name="Log Analysis", engine='openpyxl')
    
    total = len(df_summary)
    print(f"Processing {total} templates...")
    print("="*40)

    explanations = []
    
    for index, row in df_summary.iterrows():
        tpl = row['Template Pattern']
        
        # Generate
        explanation = explain_log_phi3(tpl)
        explanations.append(explanation)
        
        # --- REAL-TIME PRINTING ---
        print(f"LOG [{index+1}/{total}]: {tpl[:60]}...")
        print(f"AI MEANING:   {explanation}")
        print("-" * 40)

    df_summary['AI Explanation'] = explanations

    # Save
    output_filename = filename.replace(".xlsx", "_Phi3.xlsx")
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_logs.to_excel(writer, sheet_name='Log Analysis', index=False)
        df_summary.to_excel(writer, sheet_name='Template Summary', index=False)
        
    print(f"\nSUCCESS! Saved to: {output_filename}")

except Exception as e:
    print(f"Error: {e}")