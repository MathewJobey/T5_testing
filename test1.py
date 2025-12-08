from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(r"C:\Users\Mathe\Downloads\T5\flan_t5_logs_finetuned")
model = T5ForConditionalGeneration.from_pretrained(r"C:\Users\Mathe\Downloads\T5\flan_t5_logs_finetuned\checkpoint-339")

prompt = ("Log: sshd(pam_unix)[20898]: authentication failure; logname= uid=0 euid=0 "
          "tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root "
          "| Template: <*> authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= <*> <*>,")
print("PROMPT >>>", prompt)

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
out = model.generate(**inputs,
                     max_length=64,
                     num_beams=4,
                     early_stopping=True,
                     repetition_penalty=2.0,
                     no_repeat_ngram_size=3)
print(tokenizer.decode(out[0], skip_special_tokens=True))
