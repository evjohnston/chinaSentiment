import pandas as pd

print("===== FINAL CORPUS SUMMARY (Based on Raw + Filtered Data) =====\n")

# === 1. State of the Union (SOTU) ===
sotu_raw = pd.read_csv("1. SOTU/SOTU_CSVs/SOTU_rawdata/raw_sotu_data.csv")
sotu_full = pd.read_csv("1. SOTU/SOTU_CSVs/_full_sotu_data.csv")
sotu_summary = pd.read_csv("1. SOTU/SOTU_CSVs/coreferee_china_sentiment_summary.csv")
sotu_detailed = pd.read_csv("1. SOTU/SOTU_CSVs/coreferee_china_sentences_detailed.csv")

print("🟦 SOTU Speeches")
print(f"Total speeches in raw dataset: {sotu_raw.shape[0]}")
print(f"Total speeches once 2021-2025 was added: {sotu_full.shape[0]}")
print(f"Speeches with China mentions: {sotu_summary.shape[0]}")
print(f"Total China-related sentences: {sotu_detailed.shape[0]}")
print("Summary columns:", list(sotu_summary.columns))
print("Detailed columns:", list(sotu_detailed.columns), "\n")

# === 2. Miller Center ===
miller_raw = pd.read_csv("2. MCS/miller_center_speeches.csv")
miller_summary = pd.read_csv("2. MCS/china_sentiment_coref_summary.csv")
miller_detailed = pd.read_csv("2. MCS/china_sentences_coref_detailed.csv")

print("🟩 Miller Center Speeches")
print(f"Total speeches in raw dataset: {miller_raw.shape[0]}")
print(f"Speeches with China mentions: {miller_summary.shape[0]}")
print(f"Total China-related sentences: {miller_detailed.shape[0]}")
print("Summary columns:", list(miller_summary.columns))
print("Detailed columns:", list(miller_detailed.columns), "\n")

# === 3. NIC Publications ===
nic_raw = pd.read_csv("3. NIC/NICtexts_updated.csv")
nic_summary = pd.read_csv("3. NIC/china_sentiment_summary.csv")
nic_detailed = pd.read_csv("3. NIC/china_sentences_detailed.csv")
nic_unique_docs = nic_summary["Publication Title"].nunique()

print("🟨 NIC Publications")
print(f"Total documents in raw dataset: {nic_raw.shape[0]}")
print(f"Publications with China mentions: {nic_unique_docs}")
print(f"Total China-related sentences: {nic_detailed.shape[0]}")
print("Summary columns:", list(nic_summary.columns))
print("Detailed columns:", list(nic_detailed.columns), "\n")

# === 4. Congressional Hearings (if raw file exists) ===
try:
    hearings_raw = pd.read_csv("4. CCH/Hearings.csv")
    hearings_summary = pd.read_csv("4. CCH/Hearings_Sentiment_Summary_Merged_Clean.csv")
    hearings_detailed = pd.read_csv("4. CCH/Sentences_With_Sentiment_Merged_Clean.csv")

    print("🟥 Congressional Hearings")
    print(f"Total hearings in raw dataset: {hearings_raw.shape[0]}")
    print(f"Hearings with China mentions: {hearings_summary.shape[0]}")
    print(f"Total China-related sentences: {hearings_detailed.shape[0]}")
    print("Summary columns:", list(hearings_summary.columns))
    print("Detailed columns:", list(hearings_detailed.columns), "\n")

except FileNotFoundError:
    print("🟥 Congressional Hearings")
    print("Raw file not found. Skipping raw count. Using previous known value: 13,995")
    hearings_summary = pd.read_csv("4. CCH/Hearings_Sentiment_Summary_Merged_Clean.csv")
    hearings_detailed = pd.read_csv("4. CCH/Sentences_With_Sentiment_Merged_Clean.csv")
    print(f"Hearings with China mentions: {hearings_summary.shape[0]}")
    print(f"Total China-related sentences: {hearings_detailed.shape[0]}")
    print("Summary columns:", list(hearings_summary.columns))
    print("Detailed columns:", list(hearings_detailed.columns), "\n")

# === Create HTML summary table ===
summary_data = [
    {
        "Corpus": "SOTU",
        "Raw Documents": sotu_raw.shape[0],
        "Final Documents": sotu_full.shape[0],
        "With China Mentions": sotu_summary.shape[0],
        "China-Related Sentences": sotu_detailed.shape[0]
    },
    {
        "Corpus": "Miller Center",
        "Raw Documents": miller_raw.shape[0],
        "Final Documents": miller_raw.shape[0],
        "With China Mentions": miller_summary.shape[0],
        "China-Related Sentences": miller_detailed.shape[0]
    },
    {
        "Corpus": "NIC Publications",
        "Raw Documents": nic_raw.shape[0],
        "Final Documents": nic_raw.shape[0],
        "With China Mentions": nic_unique_docs,
        "China-Related Sentences": nic_detailed.shape[0]
    },
    {
        "Corpus": "Congressional Hearings",
        "Raw Documents": hearings_raw.shape[0] if 'hearings_raw' in locals() and not hearings_raw.empty else "N/A",
        "Final Documents": hearings_summary.shape[0],
        "With China Mentions": hearings_summary.shape[0],
        "China-Related Sentences": hearings_detailed.shape[0]
    }
]

summary_df = pd.DataFrame(summary_data)
summary_df.to_html("corpus_summary.html", index=False)
print("✅ HTML summary table saved as 'corpus_summary.html'")
