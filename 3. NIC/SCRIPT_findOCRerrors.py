import pandas as pd
import os

# Load the CSV file
df = pd.read_csv('3. NIC/NICtexts_updated.csv')

# Ensure 'Full Text' is treated as string (to avoid errors)
df['Full Text'] = df['Full Text'].astype(str)

# Filter rows with fewer than 1000 characters OR blank (empty or NaN before conversion)
filtered_df = df[(df['Full Text'].str.len() < 1000) | (df['Full Text'].str.strip() == '')]

# Display the filtered rows
print(filtered_df)

# Save to the same folder
output_path = '3. NIC/NICtexts_short_entries.csv'
filtered_df.to_csv(output_path, index=False)

print(f"Filtered data saved to: {output_path}")