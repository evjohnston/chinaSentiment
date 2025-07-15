import json
import csv
import os

def clean_text(text):
    if isinstance(text, str):
        return text.replace('\u2028', ' ').replace('\u2029', ' ').replace('\r', ' ').replace('\n', ' ').strip()
    return text

def json_to_csv(json_filename='miller_center_speeches.json', csv_filename='miller_center_speeches.csv', folder='2. MCS'):
    # Ensure the output directory exists
    os.makedirs(folder, exist_ok=True)

    json_path = os.path.join(folder, json_filename)
    csv_path = os.path.join(folder, csv_filename)

    with open(json_path, 'r', encoding='utf-8') as f:
        speeches = json.load(f)

    # Select keys to include in the CSV
    fieldnames = ['title', 'president', 'date', 'doc_name', 'transcript']

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for speech in speeches:
            writer.writerow({key: clean_text(speech.get(key, '')) for key in fieldnames})

    print(f"CSV file saved as: {csv_path}")

if __name__ == "__main__":
    json_to_csv()
