import os
import pandas as pd
import re
import requests

# === File paths ===
csv_path = "4. CCH/Hearings.csv"
txt_folder = "4. CCH/CCH_TXTfiles_per_hearing"
output_csv = "4. CCH/Hearings_Fulltext.csv"

# === Clean Text Function ===
def clean_text(text):
    text = re.sub(r'\n?Page\s*\d+\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?\s*—?\s*\d+\s*—?\s*\n?', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Load CSV ===
df = pd.read_csv(csv_path, dtype=str)
df.columns = df.columns.str.strip()
df["Full Text"] = ""

# === Insert Cleaned Text ===
for i, row in df.iterrows():
    package_id = str(row["packageId"]).strip()
    txt_filename = f"transcript_{package_id}.txt"
    txt_path = os.path.join(txt_folder, txt_filename)

    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        cleaned = clean_text(text)
        df.at[i, "Full Text"] = cleaned
        print(f"[✓] Inserted text for: {package_id}")
    else:
        print(f"[!] Missing transcript for: {package_id}")

# === Clean and Sort ===
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].notna()]
df["year"] = df["year"].astype(int)

if "htmlSizeMB" in df.columns:
    df["htmlSizeMB"] = pd.to_numeric(df["htmlSizeMB"], errors="coerce")
    df = df.sort_values("htmlSizeMB")

df["index"] = range(1, len(df) + 1)

# === Save ===
df.to_csv(output_csv, index=False)
print(f"\n✅ Final CSV saved as: {output_csv}")
print(f"📊 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"📄 Columns in final CSV: {list(df.columns)}")
