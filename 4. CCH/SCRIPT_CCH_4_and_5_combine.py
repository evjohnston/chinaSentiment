import pandas as pd
import os

# === File paths ===
folder = "4. CCH"
summary_path = os.path.join(folder, "Hearings_Sentiment_Summary.csv")
detailed_path = os.path.join(folder, "Hearings_Sentiment_Detailed.csv")
hearings_path = os.path.join(folder, "Hearings.csv")
congress_path = os.path.join(folder, "Congressional_Session_Metadata.csv")

# === Load Data ===
summary_df = pd.read_csv(summary_path)
detailed_df = pd.read_csv(detailed_path)
hearings_df = pd.read_csv(hearings_path)
congress_df = pd.read_csv(congress_path)

# Ensure date parsing
hearings_df["publishdate"] = pd.to_datetime(hearings_df["publishdate"])
congress_df["Congress Start Date"] = pd.to_datetime(congress_df["Congress Start Date"])
congress_df["Congress End Date"] = pd.to_datetime(congress_df["Congress End Date"])

# === Add publishdate to summary & detailed ===
publish_dates = hearings_df[["packageId", "publishdate"]]
summary_df = summary_df.merge(publish_dates, on="packageId", how="left")
detailed_df = detailed_df.merge(publish_dates, on="packageId", how="left")

# === Add Congress context based on publishdate range ===
def assign_congress(row, congress_table):
    for _, congress_row in congress_table.iterrows():
        if congress_row["Congress Start Date"] <= row["publishdate"] <= congress_row["Congress End Date"]:
            return congress_row
    return pd.Series([None]*len(congress_table.columns), index=congress_table.columns)

summary_congress = summary_df.apply(lambda row: assign_congress(row, congress_df), axis=1)
detailed_congress = detailed_df.apply(lambda row: assign_congress(row, congress_df), axis=1)

summary_merged = pd.concat([summary_df, summary_congress], axis=1)
detailed_merged = pd.concat([detailed_df, detailed_congress], axis=1)

# === Clean function for both datasets ===
def clean_dataframe(df, is_detailed=False):
    # Drop duplicated president column and fix names
    df = df.drop(columns=["President"], errors="ignore")
    df = df.rename(columns={"President.1": "President"})
    df["publishdate"] = pd.to_datetime(df["publishdate"], errors="coerce")

    df = df.drop(columns=[
        "Congress Start Date", "Congress End Date", 
        "President Start Date", "President End Date"
    ], errors="ignore")

    df = df.rename(columns={"Congress": "Congress Session"})

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

    df = df[ordered_cols]
    df = df.sort_values(by=["packageId", "publishdate", "Year"]).reset_index(drop=True)
    return df

# === Clean both datasets ===
summary_cleaned = clean_dataframe(summary_merged, is_detailed=False)
detailed_cleaned = clean_dataframe(detailed_merged, is_detailed=True)

# === Save cleaned datasets ===
summary_cleaned_path = os.path.join(folder, "Hearings_Sentiment_Summary_Merged_Clean.csv")
detailed_cleaned_path = os.path.join(folder, "Hearings_Sentiment_Detailed_Merged_Clean.csv")

summary_cleaned.to_csv(summary_cleaned_path, index=False)
detailed_cleaned.to_csv(detailed_cleaned_path, index=False)

# === Done ===
print("✅ Cleaned datasets saved:")
print(" -", summary_cleaned_path)
print(" -", detailed_cleaned_path)
