import pyreadr
import pandas as pd
import os

# Path to the .rda files
rda_path = os.path.join("rpkg", "sotu", "data")

# Load both R data objects
meta_result = pyreadr.read_r(os.path.join(rda_path, "sotu_meta.rda"))
text_result = pyreadr.read_r(os.path.join(rda_path, "sotu_text.rda"))

# Access by key
meta = meta_result["sotu_meta"]
text = text_result["sotu_text"]
text.columns = ['text']

# Combine metadata + speech text
df = pd.concat([meta, text], axis=1)

# Save the combined historical data
df.to_csv("rawdata/raw_sotu_data.csv", index=False)

# Load 2021–2025 data
new_data = pd.read_csv("rawdata/2021to2025_data.csv")

# Combine with existing data
combined_df = pd.concat([df, new_data], ignore_index=True)

# Save final combined dataset
combined_df.to_csv("full_sotu_data.csv", index=False)

print("Combined dataset saved as 'full_sotu_data.csv'")
