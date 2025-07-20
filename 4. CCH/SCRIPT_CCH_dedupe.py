import pandas as pd
import os

# File paths
input_path = "4. CCH/Hearings_Sentiment_Skipped.csv"
output_path = "4. CCH/Hearings_Sentiment_Skipped.csv"

# Load data
df = pd.read_csv(input_path)

# Initial count
original_count = len(df)

# Drop duplicate rows
df_no_duplicates = df.drop_duplicates()
duplicates_removed = original_count - len(df_no_duplicates)

# Drop rows where "Full Text" is missing or blank (strip whitespace)
if "Full Text" in df_no_duplicates.columns:
    df_cleaned = df_no_duplicates[df_no_duplicates["Full Text"].astype(str).str.strip() != ""]
    blank_texts_removed = len(df_no_duplicates) - len(df_cleaned)
else:
    print("⚠️ 'Full Text' column not found. Skipping blank-text removal.")
    df_cleaned = df_no_duplicates
    blank_texts_removed = 0

# Final count
final_count = len(df_cleaned)

# Save cleaned data
df_cleaned.to_csv(output_path, index=False)

# Report
print(f"✅ Cleaning Complete:")
print(f"- Original rows: {original_count}")
print(f"- Duplicates removed: {duplicates_removed}")
print(f"- Blank 'Full Text' rows removed: {blank_texts_removed}")
print(f"- Final rows: {final_count}")
print(f"💾 Cleaned file saved to: {output_path}")
