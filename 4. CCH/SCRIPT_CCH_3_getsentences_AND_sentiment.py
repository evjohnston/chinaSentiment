import os
import re
import pandas as pd
import spacy
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
from scipy.special import softmax
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

nltk.download("punkt")

# === File paths ===
folder = "4. CCH"
input_csv = os.path.join(folder, "Hearings_Fulltext.csv")
summary_csv = os.path.join(folder, "Hearings_Sentiment_Summary.csv")
detailed_csv = os.path.join(folder, "Hearings_Sentiment_Detailed.csv")
skipped_log_csv = os.path.join(folder, "Hearings_Sentiment_Skipped.csv")

# === Load data ===
df = pd.read_csv(input_csv)

# === Resume Support: track already processed packageIds
if os.path.exists(summary_csv):
    processed_ids = set(pd.read_csv(summary_csv)["packageId"].dropna().unique())
    print(f"🔄 Resuming: {len(processed_ids)} rows already processed.")
    df = df[~df["packageID"].isin(processed_ids)]
else:
    processed_ids = set()

# === China-related keywords ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan", "tibet", "xinjiang",
    "one china", " prc ", " pla ", " ccp ", "belt and road", "silk road", 
    "spy balloon", "balloon incident", 
]
china_pattern = re.compile(r"\b(" + "|".join(re.escape(k.lower()) for k in china_keywords) + r")\b")

# === Load NLP tools ===
print("🔁 Loading spaCy...")
nlp = spacy.load("en_core_web_lg")
nlp.max_length = 10_000_000

print("🔁 Loading sentiment model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# === Utility functions ===
def is_noise_sentence(text):
    text = text.strip().lower()
    return (
        text.startswith("(applause") or
        len(text) < 10 or
        re.match(r"^\(.*\)$", text)
    )

def extract_china_sentences(text):
    doc = nlp(text)
    return [
        sent.text.strip()
        for sent in doc.sents
        if china_pattern.search(sent.text.lower()) and not is_noise_sentence(sent.text)
    ]

def compute_sentiment(sentence):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = softmax(outputs.logits.numpy()[0])
        return scores[2] - scores[0]  # pos - neg
    except Exception as e:
        print(f"⚠️ Error scoring sentence: {e}")
        return 0.0

def compute_batch_sentiments(sentences, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(compute_sentiment, sentences))

# === Main processing ===
summary_records = []
sentence_records = []
skipped_rows = []
record_buffer = 0

print(f"🚀 Processing {len(df)} rows...\n")

for i, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Processing Rows", unit="row", dynamic_ncols=True):
    package_id = row.get("packageID", "")
    speech = str(row.get("text", "")).strip()

    if not speech or len(speech) < 1000:
        skipped_rows.append(row)
        continue

    china_sents = extract_china_sentences(speech)

    if china_sents:
        sentiment_scores = compute_batch_sentiments(china_sents)

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        mention_count = len(china_pattern.findall(" ".join(china_sents).lower()))

        summary_records.append({
            "packageId": package_id,
            "President": row.get("president", ""),
            "Year": row.get("year", ""),
            "Title": row.get("title", ""),
            "China Sentiment": avg_sentiment,
            "China Mention Count": mention_count
        })

        for sent, score in zip(china_sents, sentiment_scores):
            sentence_records.append({
                "packageId": package_id,
                "President": row.get("president", ""),
                "Year": row.get("year", ""),
                "Title": row.get("title", ""),
                "Sentence": sent,
                "Sentiment": score
            })

    else:
        skipped_rows.append(row)

    record_buffer += 1

    # === Save every 200 records
    if record_buffer >= 200:
        if summary_records:
            pd.DataFrame(summary_records).to_csv(summary_csv, mode='a', index=False,
                                                 header=not os.path.exists(summary_csv))
            summary_records = []

        if sentence_records:
            pd.DataFrame(sentence_records).to_csv(detailed_csv, mode='a', index=False,
                                                  header=not os.path.exists(detailed_csv))
            sentence_records = []

        if skipped_rows:
            pd.DataFrame(skipped_rows).to_csv(skipped_log_csv, mode='a', index=False,
                                              header=not os.path.exists(skipped_log_csv))
            skipped_rows = []

        record_buffer = 0

# === Final save
if summary_records:
    pd.DataFrame(summary_records).to_csv(summary_csv, mode='a', index=False,
                                         header=not os.path.exists(summary_csv))

if sentence_records:
    pd.DataFrame(sentence_records).to_csv(detailed_csv, mode='a', index=False,
                                          header=not os.path.exists(detailed_csv))

if skipped_rows:
    pd.DataFrame(skipped_rows).to_csv(skipped_log_csv, mode='a', index=False,
                                      header=not os.path.exists(skipped_log_csv))

print("\n✅ Done!")
print(f"- Summary saved: {summary_csv}")
print(f"- Detailed saved: {detailed_csv}")
if skipped_rows:
    print(f"- Skipped rows saved: {skipped_log_csv}")