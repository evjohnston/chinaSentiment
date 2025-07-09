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
plt.show()

# === Mentions over Time ===
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="china_mention_count", hue="party", marker="o")
plt.title("Mentions of China Over Time")
plt.ylabel("China Mention Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_mentions_over_time.png")
plt.show()

# === Sentiment by President ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="president", y="china_sentiment", hue="party")
plt.title("Average Sentiment Toward China by President")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_sentiment_by_president.png")
plt.show()

# === Mentions by President ===
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="president", y="china_mention_count", hue="party")
plt.title("Total Mentions of China by President")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_mentions_by_president.png")
plt.show()

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
plt.show()

# Mentions Over Time (2001–2025)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_recent, x="year", y="china_mention_count", hue="party", marker="o")
plt.title("Mentions of China (2001–2025)")
plt.ylabel("China Mention Count")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig("CHART_mentions_over_time_2001_2025.png")
plt.show()

# ========== PRESIDENT-SPECIFIC COMPARISONS ==========
# Sentiment by President
plt.figure(figsize=(12, 6))
sns.barplot(data=df_recent, x="president", y="china_sentiment", hue="party")
plt.title("Average Sentiment Toward China by President (2001-2025)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_sentiment_by_president_2001_2025.png")
plt.show()

# Mentions by President
plt.figure(figsize=(12, 6))
sns.barplot(data=df_recent, x="president", y="china_mention_count", hue="party")
plt.title("Total Mentions of China by President (2001-2025)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CHART_mentions_by_president_2001_2025.png")
plt.show()

print("✅ Visualizations saved: full range + 2001–2025 subset.")

