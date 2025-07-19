import os
import re
import pandas as pd
from datetime import datetime

# Settings
folder_path = "4. CCH/CCH_govInfo_metadata_by_year"
output_file = "4. CCH/Hearings.csv"
drop_columns = [
    'index', 'xmlLink', 'xmlSize', 'otherLink1', 'other1Size',
    'otherLink2', 'other2Size', 'historical', 'detailsLink',
    'pdfLink', 'pdfSize', 'teaser'
]

# President terms
presidents = [
    {"name": "Franklin D. Roosevelt", "start": "March 4, 1933", "end": "April 12, 1945"},
    {"name": "Harry S. Truman",       "start": "April 12, 1945", "end": "January 20, 1953"},
    {"name": "Dwight D. Eisenhower", "start": "January 20, 1953", "end": "January 20, 1961"},
    {"name": "John F. Kennedy",      "start": "January 21, 1961", "end": "November 22, 1963"},
    {"name": "Lyndon B. Johnson",    "start": "November 22, 1963", "end": "January 20, 1969"},
    {"name": "Richard Nixon",        "start": "January 21, 1969", "end": "August 9, 1974"},
    {"name": "Gerald Ford",          "start": "August 9, 1974",    "end": "January 20, 1977"},
    {"name": "Jimmy Carter",         "start": "January 21, 1977", "end": "January 20, 1981"},
    {"name": "Ronald Reagan",        "start": "January 21, 1981", "end": "January 20, 1989"},
    {"name": "George H. W. Bush",    "start": "January 21, 1989", "end": "January 20, 1993"},
    {"name": "Bill Clinton",         "start": "January 21, 1993", "end": "January 20, 2001"},
    {"name": "George W. Bush",       "start": "January 21, 2001", "end": "January 20, 2009"},
    {"name": "Barack Obama",         "start": "January 21, 2009", "end": "January 20, 2017"},
    {"name": "Donald Trump",         "start": "January 21, 2017", "end": "January 20, 2021"},
    {"name": "Joe Biden",            "start": "January 21, 2021", "end": "January 20, 2025"},
    {"name": "Donald Trump",         "start": "January 21, 2025", "end": "January 20, 2029"}
]

for p in presidents:
    p['start'] = datetime.strptime(p['start'], "%B %d, %Y")
    p['end'] = datetime.strptime(p['end'], "%B %d, %Y")

def get_president(date):
    if pd.isnull(date):
        return None
    for p in presidents:
        if p['start'] <= date <= p['end']:
            return p['name']
    return "Unknown"

def convert_to_mb(size_str):
    if pd.isnull(size_str):
        return 0.0
    match = re.match(r"([\d.]+)\s*(B|KB|MB|GB)", str(size_str).strip(), re.IGNORECASE)
    if not match:
        return 0.0
    size, unit = float(match.group(1)), match.group(2).upper()
    if unit == "B":
        return size / 1_000_000
    elif unit == "KB":
        return size / 1_000
    elif unit == "MB":
        return size
    elif unit == "GB":
        return size * 1_000
    return 0.0

dataframes = [] 

# ADDED: Track drop stats
total_drops = {"search_terms": 0, "bulk_submission": 0, "small_html": 0}

for filename in os.listdir(folder_path):
    if filename.endswith(".csv") and filename.startswith("CHearings_"):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            initial_count = len(df)

            # Search Terms / Total Count
            search_drop = df.iloc[:, 0].astype(str).str.contains("Search Terms|Total Count", na=False)
            dropped_search = search_drop.sum()
            df = df[~search_drop]

            # Bulk Submission
            dropped_bulk = 0
            if 'collection' in df.columns:
                bulk_drop = df['collection'] == 'Bulk Submission'
                dropped_bulk = bulk_drop.sum()
                df = df[~bulk_drop]

            # htmlSize < 0.01 MB
            dropped_html = 0
            if 'htmlSize' in df.columns:
                df['htmlSizeMB'] = df['htmlSize'].apply(convert_to_mb)
                small_html_drop = df['htmlSizeMB'] < 0.01
                dropped_html = small_html_drop.sum()
                df = df[~small_html_drop]

            # Drop specified columns
            df = df.drop(columns=drop_columns, errors='ignore')

            file_total_dropped = dropped_search + dropped_bulk + dropped_html
            print(f"📄 {filename} | Dropped: {file_total_dropped} rows (Search: {dropped_search}, Bulk: {dropped_bulk}, Small HTML: {dropped_html})")

            total_drops['search_terms'] += dropped_search
            total_drops['bulk_submission'] += dropped_bulk
            total_drops['small_html'] += dropped_html

            dataframes.append(df)
            print(f"✅ Loaded: {filename} with {len(df)} rows retained")

        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")

# Combine and enhance
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)

    if 'publishdate' in combined_df.columns:
        combined_df['publishdate'] = pd.to_datetime(combined_df['publishdate'], errors='coerce')
        combined_df['year'] = combined_df['publishdate'].dt.year
        combined_df['president'] = combined_df['publishdate'].apply(get_president)

        if 'htmlSizeMB' in combined_df.columns:
            combined_df = combined_df.sort_values(by='htmlSizeMB', ascending=True)

        cols = combined_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('publishdate')))
        combined_df = combined_df[cols]

        combined_df['index'] = range(1, len(combined_df) + 1)
        cols = combined_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('index')))
        combined_df = combined_df[cols]

    combined_df.to_csv(output_file, index=False)
    print(f"\n🎉 Combined, cleaned, and enriched {len(dataframes)} files into '{output_file}' with {len(combined_df)} rows.")
else:
    print("❌ No CSV files were processed.")

print(f"📊 Final DataFrame shape: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")

# ADDED: Summary of all drops
total_dropped = sum(total_drops.values())
print(f"\n🗑️ Total Records Dropped: {total_dropped} (Search Terms: {total_drops['search_terms']}, Bulk Submission: {total_drops['bulk_submission']}, Small HTML: {total_drops['small_html']})")
