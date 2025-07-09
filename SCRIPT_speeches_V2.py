import requests
import time
import json

def download_all_speeches(save_to_file='miller_center_speeches.json'):
    endpoint = "https://api.millercenter.org/speeches"
    all_speeches = []

    print("Fetching initial batch of speeches...")
    r = requests.post(url=endpoint)
    if r.status_code != 200:
        raise Exception(f"Error fetching data: {r.status_code} - {r.text}")

    data = r.json()
    all_speeches.extend(data.get('Items', []))

    # Handle pagination
    while 'LastEvaluatedKey' in data:
        doc_name = data['LastEvaluatedKey']['doc_name']
        print(f"Fetching next batch after doc_name: {doc_name}")
        params = {"LastEvaluatedKey": doc_name}
        r = requests.post(url=endpoint, params=params)
        if r.status_code != 200:
            raise Exception(f"Error fetching data: {r.status_code} - {r.text}")

        data = r.json()
        all_speeches.extend(data.get('Items', []))

        # Optional: Be polite to the API
        time.sleep(0.2)

    print(f"Downloaded {len(all_speeches)} speeches. Saving to file...")

    with open(save_to_file, 'w', encoding='utf-8') as f:
        json.dump(all_speeches, f, indent=2, ensure_ascii=False)

    print(f"Saved to {save_to_file}")

if __name__ == "__main__":
    download_all_speeches()
