import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# === Setup ===
folder = "4. CCH"
vis_folder = os.path.join(folder, "CCH_visualizations")
os.makedirs(vis_folder, exist_ok=True)

summary_path = os.path.join(folder, "Hearings_Sentiment_Summary_Merged_Clean.csv")
summary_df = pd.read_csv(summary_path, parse_dates=["publishdate"])

# === Filter to 1990–2025 ===
summary_df = summary_df[summary_df["Year"].between(1990, 2025)]

# === Group and summarize sentiment by congressional session ===
sentiment_by_session = summary_df.groupby(
    ["Congress Session", "Presidential Party", "House Majority", "Senate Majority", "President", "Party Government"]
).agg({
    "China Sentiment": "mean"
}).reset_index()

# === Construct full Party Government label ===
def combine_party_gov(row):
    pg = str(row["Party Government"]).strip()
    pres_party = str(row["Presidential Party"]).strip()
    if pg == "Divided":
        return "Divided"
    elif pg == "Unified":
        return f"Unified {pres_party}"
    else:
        return "Other"

sentiment_by_session["Party Government"] = sentiment_by_session.apply(combine_party_gov, axis=1)

# === Sort by numeric session number ===
sentiment_by_session["Session Number"] = sentiment_by_session["Congress Session"].str.extract(r'(\d+)').astype(int)
sentiment_by_session = sentiment_by_session.sort_values("Session Number")

# === Assign colors ===
party_colors = {
    "Unified Democrat": "blue",
    "Unified Republican": "red",
    "Divided": "green",
    "Other": "gray"
}
sentiment_by_session["color"] = sentiment_by_session["Party Government"].map(party_colors).fillna("gray")

# === Optional: Debug party labels ===
print("Party Government values mapped:", sentiment_by_session["Party Government"].unique())

# === Plotting ===
plt.figure(figsize=(18, 8))
bars = plt.bar(
    sentiment_by_session["Congress Session"],
    sentiment_by_session["China Sentiment"],
    color=sentiment_by_session["color"]
)

# Add president name inside the bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    pres = sentiment_by_session.iloc[i]["President"]
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height / 2,
        pres,
        ha="center",
        va="center",
        fontsize=9,
        color="white" if abs(height) > 0.05 else "black",
        rotation=90
    )

# === Add Trend Line ===
x = sentiment_by_session["Session Number"]
y = sentiment_by_session["China Sentiment"]
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(
    sentiment_by_session["Congress Session"],
    p(x),
    color="black",
    linestyle="--",
    linewidth=2,
    label="Trend Line"
)

# Labels and title
plt.title("Normalized Avg Sentiment Toward China by Congressional Session (1990–2025)")
plt.xlabel("Congressional Session")
plt.ylabel("Average Sentiment per Hearing")
plt.xticks(rotation=45, ha="right")

# Legend
legend_patches = [
    mpatches.Patch(color='blue', label='Unified Democrat'),
    mpatches.Patch(color='red', label='Unified Republican'),
    mpatches.Patch(color='green', label='Divided'),
    mpatches.Patch(color='gray', label='Other'),
    mpatches.Patch(color='black', label='Trend Line')
]
plt.legend(handles=legend_patches, title="Party Government")

# Save plot
output_path = os.path.join(vis_folder, "normalized_avg_sentiment_by_congress_session.png")
plt.tight_layout()
plt.savefig(output_path)
print(f"✅ Saved bar chart to {output_path}")
