import pandas as pd
import os

# === File paths ===
folder = "4. CCH"
summary_path = os.path.join(folder, "Hearings_Sentiment_Summary_Merged.csv")
detailed_path = os.path.join(folder, "Hearings_Sentiment_Detailed_Merged.csv")

# === Load data ===
summary_df = pd.read_csv(summary_path)
detailed_df = pd.read_csv(detailed_path)

# === Clean function for both datasets ===
def clean_dataframe(df, is_detailed=False):
    # Drop the duplicate president column and rename the correct one
    df = df.drop(columns=["President"])
    df = df.rename(columns={"President.1": "President"})

    # Convert publishdate to datetime
    df["publishdate"] = pd.to_datetime(df["publishdate"], errors="coerce")

    # Drop unused congress/president date ranges
    df = df.drop(columns=[
        "Congress Start Date", "Congress End Date", 
        "President Start Date", "President End Date"
    ])

    # Rename column
    df = df.rename(columns={"Congress": "Congress Session"})

    # Define final sort order
    base_order = [
        "packageId", "publishdate", "Year", "Title", "President", 
        "Presidential Party", "Congress Session", "House Majority", 
        "Senate Majority", "Party Government"
    ]

    detailed_fields = ["Sentence", "Sentiment"]
    summary_fields = ["China Mention Count", "China Sentiment"]

    ordered_cols = (
        base_order[:4] + detailed_fields + base_order[4:] if is_detailed 
        else base_order[:4] + summary_fields + base_order[4:]
    )

    # Reorder and sort
    df = df[ordered_cols]
    df = df.sort_values(by=["packageId", "publishdate", "Year"]).reset_index(drop=True)

    return df

# === Clean both datasets ===
summary_cleaned = clean_dataframe(summary_df, is_detailed=False)
detailed_cleaned = clean_dataframe(detailed_df, is_detailed=True)

# === Save cleaned datasets ===
summary_cleaned_path = os.path.join(folder, "Hearings_Sentiment_Summary_Merged_Clean.csv")
detailed_cleaned_path = os.path.join(folder, "Hearings_Sentiment_Detailed_Merged_Clean.csv")

summary_cleaned.to_csv(summary_cleaned_path, index=False)
detailed_cleaned.to_csv(detailed_cleaned_path, index=False)

print("✅ Cleaned datasets saved:")
print(" -", summary_cleaned_path)
print(" -", detailed_cleaned_path)

# === Sentiment Summary Data ===
print("\n=== Sentiment Summary Data ===")
print("Shape:", summary_cleaned.shape)
print("Columns:", summary_cleaned.columns.tolist())
print("\nInfo:")
print(summary_cleaned.info())
print("\nHead:\n", summary_cleaned.head())

# === Detailed Sentences Data ===
print("\n=== Detailed Sentences Data ===")
print("Shape:", detailed_cleaned.shape)
print("Columns:", detailed_cleaned.columns.tolist())
print("\nInfo:")
print(detailed_cleaned.info())
print("\nHead:\n", detailed_cleaned.head())
