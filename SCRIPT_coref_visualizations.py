# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("CSV_china_sentiment_coref_summary.csv")

# Sort for time series
df = df.sort_values("year")

# Set seaborn style
sns.set(style="whitegrid", font_scale=1.2)

# === Sentiment over Time ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="china_sentiment", hue="party", marker="o")
plt.title("Sentiment Toward China Over Time")
plt.ylabel("Average Sentiment")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_sentiment_over_time.png")

# === Mentions over Time ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="china_mention_count", hue="party", marker="o")
plt.title("Mentions of China Over Time")
plt.ylabel("China Mention Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_mentions_over_time.png")

# === Sentiment by President ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="president", y="china_sentiment", hue="party")
plt.title("Average Sentiment Toward China by President")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_sentiment_by_president.png")

# === Mentions by President ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="president", y="china_mention_count", hue="party")
plt.title("Total Mentions of China by President")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_mentions_by_president.png")

# ========== 2001–2025 SUBSET VISUALIZATIONS ==========
df_recent = df[(df["year"] >= 2001) & (df["year"] <= 2025)]

# Sentiment Over Time (2001–2025)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_recent, x="year", y="china_sentiment", hue="party", marker="o")
plt.title("Sentiment Toward China (2001–2025)")
plt.ylabel("Average Sentiment")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_sentiment_over_time_2001_2025.png")

# Mentions Over Time (2001–2025)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_recent, x="year", y="china_mention_count", hue="party", marker="o")
plt.title("Mentions of China (2001–2025)")
plt.ylabel("China Mention Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_mentions_over_time_2001_2025.png")

# ========== PRESIDENT-SPECIFIC COMPARISONS ==========
# Sentiment by President
plt.figure(figsize=(12, 6))
sns.barplot(data=df_recent, x="president", y="china_sentiment", hue="party")
plt.title("Average Sentiment Toward China by President (2001-2025)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_sentiment_by_president_2001_2025.png")

# Mentions by President
plt.figure(figsize=(12, 6))
sns.barplot(data=df_recent, x="president", y="china_mention_count", hue="party")
plt.title("Total Mentions of China by President (2001-2025)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_mentions_by_president_2001_2025.png")

print("✅ Visualizations saved: full range + 2001–2025 subset.")

# === Sentiment over Time (no party hue) ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="china_sentiment", marker="o")
plt.title("Sentiment Toward China Over Time")
plt.ylabel("Average Sentiment")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_sentiment_over_time_NO_PARTY.png")

# === Mentions over Time (no party hue) ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="china_mention_count", marker="o")
plt.title("Mentions of China Over Time")
plt.ylabel("China Mention Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_mentions_over_time_NO_PARTY.png")

# === Sentiment by President (no party hue) ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="president", y="china_sentiment")
plt.title("Average Sentiment Toward China by President")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("CHART_sentiment_by_president_NO_PARTY.png")

# === Mentions by President (no party hue) ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="president", y="china_mention_count")
plt.title("Total Mentions of China by President")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("CHART_mentions_by_president_NO_PARTY.png")

# === Sentiment Over Time (2001–2025, no party) ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_recent, x="year", y="china_sentiment", marker="o")
plt.title("Sentiment Toward China (2001–2025)")
plt.ylabel("Average Sentiment")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_sentiment_over_time_2001_2025_NO_PARTY.png")

# === Mentions Over Time (2001–2025, no party) ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_recent, x="year", y="china_mention_count", marker="o")
plt.title("Mentions of China (2001–2025)")
plt.ylabel("China Mention Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_mentions_over_time_2001_2025_NO_PARTY.png")

# === Sentiment by President (2001–2025, no party) ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df_recent, x="president", y="china_sentiment", estimator="mean")
plt.title("Average Sentiment Toward China by President (2001–2025)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("CHART_sentiment_by_president_2001_2025_NO_PARTY.png")

# === Mentions by President (2001–2025, no party) ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df_recent, x="president", y="china_mention_count", estimator="sum")
plt.title("Total Mentions of China by President (2001–2025)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("CHART_mentions_by_president_2001_2025_NO_PARTY.png")
