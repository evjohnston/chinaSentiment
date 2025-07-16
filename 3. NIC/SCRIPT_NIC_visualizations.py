import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud

# === Load the data ===
folder = "3. NIC"
visuals_folder = os.path.join(folder, "visualizations")
os.makedirs(visuals_folder, exist_ok=True)

# === Load summary and detailed sentence-level data ===
summary_file = os.path.join(folder, "china_sentiment_summary.csv")
detailed_file = os.path.join(folder, "china_sentences_detailed.csv")
df = pd.read_csv(summary_file)
df_sentences = pd.read_csv(detailed_file)

# === Normalize headers ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df_sentences.columns = df_sentences.columns.str.strip().str.lower().str.replace(" ", "_")

# === Convert data types safely ===
for col in ["china_sentiment", "china_mention_count", "year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["china_sentiment", "china_mention_count", "year"])

for col in ["sentiment", "year"]:
    df_sentences[col] = pd.to_numeric(df_sentences[col], errors="coerce")

df_sentences = df_sentences.dropna(subset=["sentiment", "year"])

# === Sort for time series ===
df = df.sort_values("year")

# === Set seaborn style ===
sns.set(style="whitegrid", font_scale=1.2)

# === Sentiment by President ===
agg_sentiment = df.groupby(["president", "year"], as_index=False)["china_sentiment"].mean()
agg_sentiment = agg_sentiment.sort_values("year")
plt.figure(figsize=(12, 6))
sns.barplot(data=agg_sentiment, x="president", y="china_sentiment")
plt.title("Average Sentiment Toward China by President")
plt.xticks(rotation=45, ha="right")
plt.ylim(-1, 1)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentiment_by_president.png"))

# === Mentions by President ===
agg_mentions = df.groupby(["president", "year"], as_index=False)["china_mention_count"].sum()
agg_mentions = agg_mentions.sort_values("year")
plt.figure(figsize=(12, 6))
sns.barplot(data=agg_mentions, x="president", y="china_mention_count")
plt.title("Total Mentions of China by President")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, agg_mentions["china_mention_count"].max() + 5)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_mentions_by_president.png"))

# === Subset for 2001–2025 ===
df_recent = df[(df["year"] >= 2001) & (df["year"] <= 2025)]

# === Sentiment Over Time (2001–2025) ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_recent, x="year", y="china_sentiment", marker="o")
plt.title("Sentiment Toward China Over Time (2001–2025)")
plt.ylabel("Average Sentiment")
plt.xlabel("Year")
plt.ylim(-1, 1)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentiment_over_time_2001_2025.png"))

# === Sentiment by President (2001–2025) ===
agg_recent_sentiment = df_recent.groupby(["president", "year"], as_index=False)["china_sentiment"].mean()
agg_recent_sentiment = agg_recent_sentiment.sort_values("year")
plt.figure(figsize=(12, 6))
sns.barplot(data=agg_recent_sentiment, x="president", y="china_sentiment")
plt.title("Average Sentiment Toward China by President (2001–2025)")
plt.xticks(rotation=45, ha="right")
plt.ylim(-1, 1)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentiment_by_president_2001_2025.png"))

# === Mentions by President (2001–2025) ===
agg_recent_mentions = df_recent.groupby(["president", "year"], as_index=False)["china_mention_count"].sum()
agg_recent_mentions = agg_recent_mentions.sort_values("year")
plt.figure(figsize=(12, 6))
sns.barplot(data=agg_recent_mentions, x="president", y="china_mention_count")
plt.title("Total Mentions of China by President (2001–2025)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, agg_recent_mentions["china_mention_count"].max() + 5)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_mentions_by_president_2001_2025.png"))

# === Additional Visualizations ===

# 1. Sentiment Distribution Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df_sentences["sentiment"], bins=40, kde=True)
plt.title("Distribution of Sentiment Scores for China-Related Sentences")
plt.xlabel("Sentiment Score")
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentiment_distribution_histogram.png"))

# 2. Sentiment Range by President (Boxplot)
plt.figure(figsize=(14, 6))
sns.boxplot(data=df_sentences, x="president", y="sentiment")
plt.title("Sentiment Range Toward China by President")
plt.xticks(rotation=45, ha="right")
plt.ylim(-1, 1)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentiment_range_by_president.png"))

# 3. Sentence Count per Year
sentence_count = df_sentences.groupby("year").size()
plt.figure(figsize=(12, 6))
sentence_count.plot(kind="bar")
plt.title("Number of China-Related Sentences per Year")
plt.ylabel("Sentence Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentence_count_per_year.png"))

# 4. Sentiment Polarity Over Time
df_sentences["polarity"] = df_sentences["sentiment"].apply(
    lambda x: "positive" if x > 0.1 else "negative" if x < -0.1 else "neutral"
)
plt.figure(figsize=(14, 6))
sns.countplot(data=df_sentences, x="year", hue="polarity")
plt.title("Polarity of China-Related Sentences Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_polarity_over_time.png"))

# 5. Word Cloud of China-Related Sentences
text = " ".join(df_sentences["sentence"].dropna().astype(str))
wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
wordcloud.to_file(os.path.join(visuals_folder, "CHART_wordcloud_china_sentences.png"))

# 6. Sentiment vs. Mention Count (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="china_mention_count", y="china_sentiment", hue="president")
plt.title("Sentiment vs. Mention Count by President")
plt.xlabel("China Mention Count")
plt.ylabel("Average Sentiment")
plt.tight_layout()
plt.savefig(os.path.join(visuals_folder, "CHART_sentiment_vs_mentions.png"))

print("\n✅ All core and advanced visualizations saved to:", visuals_folder)
