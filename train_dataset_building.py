import pandas as pd
our_df=pd.read_csv('logs_with_meaning.csv')
clean_df = our_df.iloc[:1000].copy()

clean_df["input_text"] = (
    "Log: " + clean_df["raw_log"].astype(str) +
    " | Template: " + clean_df["template"].astype(str)
)
clean_df["target_text"] = clean_df["meaning"]
clean_df[["input_text", "target_text"]].to_csv('tuning_dataset.csv', index=False)
print(clean_df.head())


df = pd.read_csv("tuning_dataset.csv")
print(df.info())
print(df.head(3))



