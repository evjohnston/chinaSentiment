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

# === Step 1: Add publishdate to summary & detailed ===
publish_dates = hearings_df[["packageId", "publishdate"]]
summary_df = summary_df.merge(publish_dates, on="packageId", how="left")
detailed_df = detailed_df.merge(publish_dates, on="packageId", how="left")

# === Step 2: Add Congress context based on publishdate range ===
def assign_congress(row, congress_table):
    for _, congress_row in congress_table.iterrows():
        if congress_row["Congress Start Date"] <= row["publishdate"] <= congress_row["Congress End Date"]:
            return congress_row
    return pd.Series([None]*len(congress_table.columns), index=congress_table.columns)

summary_congress = summary_df.apply(lambda row: assign_congress(row, congress_df), axis=1)
detailed_congress = detailed_df.apply(lambda row: assign_congress(row, congress_df), axis=1)

# === Step 3: Combine final data ===
summary_final = pd.concat([summary_df, summary_congress], axis=1)
detailed_final = pd.concat([detailed_df, detailed_congress], axis=1)

# === Step 4: Save merged datasets ===
summary_final.to_csv(os.path.join(folder, "Hearings_Sentiment_Summary_Merged.csv"), index=False)
detailed_final.to_csv(os.path.join(folder, "Hearings_Sentiment_Detailed_Merged.csv"), index=False)

print("✅ Merged datasets saved.")