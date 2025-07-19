import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# === Create output directory ===
output_dir = "macro_vis"
os.makedirs(output_dir, exist_ok=True)

# === Load datasets ===
sotu = pd.read_csv("1. SOTU/SOTU_CSVs/coreferee_china_sentiment_summary.csv")
miller = pd.read_csv("2. MCS/china_sentiment_coref_summary.csv")
nic = pd.read_csv("3. NIC/china_sentiment_summary.csv")
hearings = pd.read_csv("4. CCH/Hearings_Sentiment_Summary_Merged_Clean.csv")
session_meta = pd.read_csv("4. CCH/Congressional_Session_Metadata.csv")

# === Standardize sentiment data ===

# --- SOTU (uses president column directly) ---
sotu_df = sotu[['year', 'china_sentiment', 'president']].rename(columns={
    'year': 'Year',
    'china_sentiment': 'Sentiment',
    'president': 'President'
})
sotu_df['Corpus'] = 'SOTU'

# --- Miller Center ---
miller_df = miller[['year', 'china_sentiment']].rename(columns={
    'year': 'Year',
    'china_sentiment': 'Sentiment'
})
miller_df['Corpus'] = 'Miller Center'

# --- NIC Publications ---
nic_df = nic[['Year', 'China Sentiment']].rename(columns={'China Sentiment': 'Sentiment'})
nic_df['Corpus'] = 'NIC Publications'

# --- Congressional Hearings ---
hearings_df = hearings[['Year', 'China Sentiment']].rename(columns={'China Sentiment': 'Sentiment'})
hearings_df['Corpus'] = 'Congressional Hearings'

# === Combine all sources (except president labeling for SOTU) ===
combined = pd.concat([sotu_df[['Year', 'Sentiment', 'Corpus', 'President']],
                      miller_df, nic_df, hearings_df],
                     ignore_index=True)

# Filter and clean
combined.dropna(subset=['Year', 'Sentiment'], inplace=True)
combined['Year'] = combined['Year'].astype(int)
combined = combined[(combined['Year'] >= 2000) & (combined['Year'] <= 2025)]

# === Build president-year mapping from session metadata for non-SOTU corpora ===
session_meta = session_meta[['President', 'President Start Date', 'President End Date']].drop_duplicates()

def expand_presidency_rows(row):
    start = pd.to_datetime(row['President Start Date'], errors='coerce')
    end = pd.to_datetime(row['President End Date'], errors='coerce')
    if pd.isna(start) or pd.isna(end):
        return []
    return [{'Year': y, 'President': row['President']} for y in range(start.year, end.year + 1)]

expanded_years = []
for _, row in session_meta.iterrows():
    expanded_years.extend(expand_presidency_rows(row))

president_year_map = pd.DataFrame(expanded_years)

# === Assign presidents only to non-SOTU rows ===
non_sotu_mask = combined['Corpus'] != 'SOTU'
combined.loc[non_sotu_mask, 'President'] = combined[non_sotu_mask].merge(
    president_year_map, on='Year', how='left')['President_y']

combined = combined.dropna(subset=['President'])

# === Assign custom president term labels ===
def custom_president_label(row):
    if row['Year'] < 2001:
        return "William J. Clinton (2000)"
    elif 2001 <= row['Year'] <= 2008:
        return "George W. Bush (2001–2008)"
    elif 2009 <= row['Year'] <= 2016:
        return "Barack Obama (2009–2016)"
    elif 2017 <= row['Year'] <= 2020:
        return "Donald J. Trump (2017–2020)"
    elif 2021 <= row['Year'] <= 2024:
        return "Joseph R. Biden, Jr. (2021–2024)"
    elif row['Year'] >= 2025:
        return "Donald J. Trump (2025–)"
    else:
        return None

combined['President_Label'] = combined.apply(custom_president_label, axis=1)
combined = combined.dropna(subset=['President_Label'])

# === Define display order ===
pres_order = [
    "William J. Clinton (2000)",
    "George W. Bush (2001–2008)",
    "Barack Obama (2009–2016)",
    "Donald J. Trump (2017–2020)",
    "Joseph R. Biden, Jr. (2021–2024)",
    "Donald J. Trump (2025–)"
]
combined['President_Label'] = pd.Categorical(combined['President_Label'], categories=pres_order, ordered=True)

# ==================== PLOT 1.1: Line Plot ==================== #
plt.figure(figsize=(14, 7))
for corpus in combined['Corpus'].unique():
    subset = combined[combined['Corpus'] == corpus]
    yearly_avg = subset.groupby('Year')['Sentiment'].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, label=corpus)

plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
plt.title("Figure 1.1: Average Sentiment Toward China Over Time by Corpus", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Sentiment Score", fontsize=12)
plt.legend(title="Corpus")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(2000, 2025)
plt.tight_layout()
fig1_path = os.path.join(output_dir, "figure_1_1_china_sentiment_over_time_2000_2025.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"✅ Figure 1.1 saved to: {fig1_path}")

# ==================== PLOT 1.2: Boxplot by President-Term ==================== #
plt.figure(figsize=(13, 8))
sns.boxplot(data=combined, x='President_Label', y='Sentiment', hue='Corpus', palette='Set2')
plt.title("Figure 1.2: Sentiment Toward China by Presidential Term (2000–2025)", fontsize=16)
plt.xlabel("President (Term)", fontsize=12)
plt.ylabel("Sentiment Score", fontsize=12)
plt.xticks(rotation=15)
plt.legend(title="Corpus", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
fig2_path = os.path.join(output_dir, "figure_1_2_sentiment_boxplot_president_2000_2025.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"✅ Figure 1.2 saved to: {fig2_path}")

# ==================== TABLE: Document Counts by President-Term and Corpus ==================== #
doc_counts = combined.groupby(['President_Label', 'Corpus']).size().unstack(fill_value=0)
doc_counts['Total'] = doc_counts.sum(axis=1)
doc_counts = doc_counts.reindex(pres_order)

# Save to HTML
html_table_path = os.path.join(output_dir, "president_term_document_counts.html")
doc_counts.reset_index(inplace=True)
doc_counts.to_html(html_table_path, index=False)
print(f"✅ Document counts table saved to: {html_table_path}")

# ==================== PLOT 1.3: Density Plot or Histogram of Sentiment Scores ==================== #
plt.figure(figsize=(12, 7))

# Toggle between KDE (smooth density) and histogram by changing this flag
use_kde = True

if use_kde:
    sns.kdeplot(data=combined, x="Sentiment", hue="Corpus", fill=True, common_norm=False, alpha=0.4)
    plt.title("Figure 1.3: Density Plot of Sentiment Scores Toward China by Corpus", fontsize=16)
else:
    sns.histplot(data=combined, x="Sentiment", hue="Corpus", bins=30, multiple="stack", kde=False)
    plt.title("Figure 1.3: Histogram of Sentiment Scores Toward China by Corpus", fontsize=16)

plt.xlabel("Sentiment Score", fontsize=12)
plt.ylabel("Density" if use_kde else "Document Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

fig3_path = os.path.join(output_dir, "figure_1_3_sentiment_distribution.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"✅ Figure 1.3 saved to: {fig3_path}")

# ==================== PLOT 1.4: Normalized Document Count by Corpus ==================== #
from sklearn.preprocessing import MinMaxScaler

# Recalculate document counts
mention_counts = combined.groupby(['Year', 'Corpus']).size().reset_index(name='Doc Count')

# Normalize document counts within each corpus
mention_counts['Normalized Count'] = mention_counts.groupby('Corpus')['Doc Count'].transform(
    lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten()
)

# Plot
plt.figure(figsize=(14, 7))
sns.lineplot(data=mention_counts, x='Year', y='Normalized Count', hue='Corpus', marker='o')
plt.title("Figure 1.4: Normalized Document Mentions of China by Corpus (2000–2025)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Normalized Document Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

fig4_path = os.path.join(output_dir, "figure_1_4_normalized_china_mentions_by_corpus.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"✅ Figure 1.4 saved to: {fig4_path}")

# ==================== PLOT 1.5: Sentiment vs. Volume Bubble Plot ==================== #
bubble_data = combined.groupby(['Year', 'Corpus']).agg(
    Avg_Sentiment=('Sentiment', 'mean'),
    Doc_Count=('Sentiment', 'size')
).reset_index()

plt.figure(figsize=(14, 8))
sns.scatterplot(
    data=bubble_data,
    x='Year',
    y='Avg_Sentiment',
    size='Doc_Count',
    hue='Corpus',
    alpha=0.6,
    palette='Set2',
    sizes=(20, 400)
)
plt.title("Figure 1.5: Average Sentiment vs Document Volume by Corpus", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Sentiment Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Corpus", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

fig5_path = os.path.join(output_dir, "figure_1_5_sentiment_vs_volume_bubble.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"✅ Figure 1.5 saved to: {fig5_path}")

# ==================== PLOT 1.6: Stacked Area Chart of Document Counts ==================== #
doc_counts_yearly = combined.groupby(['Year', 'Corpus']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 7))
doc_counts_yearly.plot.area(stacked=True, cmap='Set2', alpha=0.85, ax=plt.gca())
plt.title("Figure 1.6: Stacked Area Chart of Document Mentions of China by Corpus", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Document Count", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

fig6_path = os.path.join(output_dir, "figure_1_6_stacked_area_chart_china_mentions.png")
plt.savefig(fig6_path, dpi=300)
plt.close()
print(f"✅ Figure 1.6 saved to: {fig6_path}")
