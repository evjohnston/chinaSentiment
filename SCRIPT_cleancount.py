import pandas as pd

# Load the CSV file
file_path = 'CSV_china_sentiment_summary.csv'
df = pd.read_csv(file_path)

# Remove rows where china_mention_count is 0
df_cleaned = df[df["china_mention_count"] != 0]

# Save the cleaned data back to the same file
df_cleaned.to_csv(file_path, index=False)

print("Cleaned data saved back to", file_path)
