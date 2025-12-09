import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tuning_dataset2.csv")

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
