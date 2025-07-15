import requests
import time
import json
import os

def clean_text_fields(speech):
    for key in ['title', 'president', 'date', 'doc_name', 'transcript']:
        if key in speech and isinstance(speech[key], str):
            speech[key] = speech[key]\
                .replace('\u2028', ' ')\
                .replace('\u2029', ' ')\
                .replace('\r', ' ')\
                .replace('\n', ' ')\
                .strip()
    return speech

def download_all_speeches(output_dir='2. MCS', filename='miller_center_speeches.json'):
    endpoint = "https://api.millercenter.org/speeches"
    all_speeches = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    save_to_file = os.path.join(output_dir, filename)

    print("Fetching initial batch of speeches...")
    r = requests.post(url=endpoint)
    if r.status_code != 200:
        raise Exception(f"Error fetching data: {r.status_code} - {r.text}")

    data = r.json()
    cleaned_items = [clean_text_fields(s) for s in data.get('Items', [])]
    all_speeches.extend(cleaned_items)

    # Handle pagination
    while 'LastEvaluatedKey' in data:
        doc_name = data['LastEvaluatedKey']['doc_name']
        print(f"Fetching next batch after doc_name: {doc_name}")
        params = {"LastEvaluatedKey": doc_name}
        r = requests.post(url=endpoint, params=params)
        if r.status_code != 200:
            raise Exception(f"Error fetching data: {r.status_code} - {r.text}")

        data = r.json()
        cleaned_items = [clean_text_fields(s) for s in data.get('Items', [])]
        all_speeches.extend(cleaned_items)

        time.sleep(0.2)

    print(f"Downloaded {len(all_speeches)} speeches. Saving to {save_to_file}...")

    with open(save_to_file, 'w', encoding='utf-8') as f:
        json.dump(all_speeches, f, indent=2, ensure_ascii=False)

    print("Download and save complete.")

if __name__ == "__main__":
    download_all_speeches()
