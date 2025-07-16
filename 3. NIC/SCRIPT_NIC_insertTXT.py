import fitz  # PyMuPDF
import os
import pandas as pd
import re

# === Paths ===
pdf_folder = "PDFs"
txt_folder = "extracted_texts"
csv_path = "NICtexts.csv"
output_csv = "NICtexts_updated.csv"

# === Ensure output folder exists ===
os.makedirs(txt_folder, exist_ok=True)

# === Clean Text Function ===
def clean_text(text):
    text = re.sub(r'\n?Page\s*\d+\n?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n?\s*—?\s*\d+\s*—?\s*\n?', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === STEP 1: Convert PDFs → cleaned TXT files ===
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)

        # Normalize name
        base_name = os.path.splitext(filename)[0].strip().lower()
        txt_filename = base_name + ".txt"
        txt_path = os.path.join(txt_folder, txt_filename)

        # Extract and clean text
        with fitz.open(pdf_path) as doc:
            raw_text = "\n".join(page.get_text() for page in doc)

        cleaned = clean_text(raw_text)

        # Save to .txt
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        print(f"[✓] Saved cleaned text to: {txt_path}")

# === STEP 2: Load CSV and inject cleaned text ===
df = pd.read_csv(csv_path)

# Ensure "Full Text" column exists
if "Full Text" not in df.columns:
    df["Full Text"] = ""

for i, row in df.iterrows():
    short_title = os.path.splitext(str(row["Short PubTitle"]).strip())[0].lower()
    txt_filename = short_title + ".txt"
    txt_path = os.path.join(txt_folder, txt_filename)

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        df.at[i, "Full Text"] = text
        print(f"[✓] Inserted text into CSV for: {short_title}")
    else:
        print(f"[!] No TXT found for: {short_title}")

# === Save Updated CSV ===
df.to_csv(output_csv, index=False)
print(f"\n✅ Final CSV saved to: {output_csv}")
