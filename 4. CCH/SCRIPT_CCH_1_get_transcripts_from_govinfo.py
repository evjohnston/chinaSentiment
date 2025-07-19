import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Config ===
csv_path = "4. CCH/Hearings.csv"
output_dir = "4. CCH/CCH_TXTfiles_per_hearing"
max_threads = 10  # You can tune this (10–20 is usually safe)

# === Setup ===
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path)
df["packageId"] = df["packageId"].astype(str).str.strip()

# === Download Function ===
def download_transcript(row):
    html_url = row.get("htmlLink")
    package_id = row.get("packageId")
    filename = f"transcript_{package_id}.txt"
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return f"[✓] Already exists: {filename}"

    try:
        response = requests.get(html_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        return f"[✓] Downloaded: {filename}"
    except Exception as e:
        return f"[!] Failed: {package_id} — {str(e)}"

# === Multithreaded Execution ===
print(f"🚀 Starting parallel download with {max_threads} threads...")
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(download_transcript, row) for _, row in df.iterrows()]
    for future in as_completed(futures):
        print(future.result())

print("✅ All done!")
