import pandas as pd

df = pd.read_csv('logs_with_meaning.csv')
clean_df = df.iloc[:1000].copy()   # use the same 1000 labeled samples

clean_df["input_text"] = "Log: " + clean_df["raw_log"].astype(str)
clean_df["target_text"] = clean_df["meaning"]

clean_df[["input_text", "target_text"]].to_csv("tuning_dataset_rawonly.csv", index=False)
print(clean_df.head())
df = pd.read_csv("tuning_dataset_rawonly.csv")