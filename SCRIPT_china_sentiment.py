# %%
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # For sentence tokenization

# %%
# === Load full SOTU dataset ===
df = pd.read_csv("CSV_full_sotu_data.csv")

# === Define China-related keywords and pronouns ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan"
]

china_pronouns = ["they", "them", "their", "theirs"]  # Optional, for deeper NLP

# Normalize all text
df["text_lower"] = df["text"].fillna("").str.lower()

# %%

# === Extract sentences mentioning China and compute sentiment ===
def extract_china_sentences(text):
    if pd.isna(text):
        return []
    sentences = sent_tokenize(text)
    return [s for s in sentences if any(k in s.lower() for k in china_keywords)]

def sentence_sentiment(sentences):
    if not sentences:
        return None
    sentiments = [TextBlob(s).sentiment.polarity for s in sentences]
    return sum(sentiments) / len(sentiments)

df["china_sentences"] = df["text"].apply(extract_china_sentences)
df["china_sentiment"] = df["china_sentences"].apply(sentence_sentiment)

# Count total mentions of China-related keywords (not per speech, but total appearances)
def total_mentions(sentences):
    joined = " ".join(sentences).lower()
    return sum(len(re.findall(r"\b" + re.escape(k) + r"\b", joined)) for k in china_keywords)

df["china_mention_count"] = df["china_sentences"].apply(total_mentions)

# %%
# === Aggregate by president, year, party ===
summary = df.groupby(["president", "year", "party"], as_index=False).agg({
    "china_sentiment": "mean",
    "china_mention_count": "sum"
})

# Save processed data
summary.to_csv("china_sentiment_summary.csv", index=False)

# %%
# %%
# === Plot 1 (Improved): Sentiment About China (sentence-level, clean x-axis) ===
plt.figure(figsize=(16, 6))

summary = summary.sort_values("year")
summary["year_label"] = summary["year"].astype(float)

# Label only first year of each president
summary["president_label"] = ""
summary.loc[
    summary["president"] != summary["president"].shift(1),
    "president_label"
] = summary["president"] + " (" + summary["year_label"].astype(str) + ")"

# Plot
sns.lineplot(data=summary, x="year_label", y="china_sentiment", hue="party", marker="o")

# Format xticks with clean labels
plt.xticks(
    ticks=summary["year_label"],
    labels=[
        label if label != "" else ""
        for label in summary["president_label"]
    ],
    rotation=90,
    fontsize=7
)

plt.title("Sentiment About China (Sentence-Level – Only Where China is Mentioned)")
plt.xlabel("President (First Year of Term)")
plt.ylabel("Average Sentiment Polarity")
plt.tight_layout()
plt.savefig("CHART_china_sentiment_sentence_level.png")
plt.close()

# %%
# === Plot 2 (Final Fix): Total Mentions of China-Related Terms by Year (discrete x-axis) ===
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 6))

summary = summary.sort_values("year")
summary["year_label"] = summary["year"].astype(float)
summary["president_label"] = ""
summary.loc[
    summary["president"] != summary["president"].shift(1),
    "president_label"
] = summary["president"] + " (" + summary["year_label"].astype(str) + ")"

# Use categorical axis: convert year_label to string for proper spacing
summary["year_str"] = summary["year_label"].astype(str)

sns.barplot(
    data=summary,
    x="year_str",
    y="china_mention_count",
    hue="party",
    dodge=False
)

# Clean x-axis: show only selected president-year labels
plt.xticks(
    ticks=range(len(summary)),
    labels=[
        lbl if lbl != "" else "" for lbl in summary["president_label"]
    ],
    rotation=90,
    fontsize=7
)

plt.title("Total Mentions of China-Related Terms by Year")
plt.xlabel("President (First Year of Term)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("CHART_china_mentions_total_by_year.png")
plt.close()

# %%
print("✅ All done (v2)! Outputs:")
print("- china_sentiment_summary_v2.csv")
print("- china_sentiment_sentence_level.png")
print("- china_mentions_total_by_year.png")
