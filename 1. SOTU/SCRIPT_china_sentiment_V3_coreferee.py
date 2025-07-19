# %%
import pandas as pd
import spacy
import coreferee
import nltk
import re
import os
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Download necessary NLTK resources
nltk.download("punkt")

# %%
# === File Paths ===
folder = "1. SOTU/SOTU_CSVs"
input_csv = os.path.join(folder, "_full_sotu_data.csv")
summary_csv = os.path.join(folder, "coreferee_china_sentiment_summary.csv")
detailed_csv = os.path.join(folder, "coreferee_china_sentences_detailed.csv")
skipped_log_csv = os.path.join(folder, "skipped_rows_due_to_invalid_dates.csv")

# === Load data ===
df = pd.read_csv(input_csv)
df["text"] = df["text"].fillna("")

# === Parse and Normalize Dates ===
df["year"] = pd.to_numeric(df["year"], errors="coerce")
invalid_dates = df[df["year"].isna()]
if not invalid_dates.empty:
    print(f"⚠️ Skipping {len(invalid_dates)} rows with invalid years.")
    invalid_dates.to_csv(skipped_log_csv, index=False)

df = df[df["year"].notna()]
df["year"] = df["year"].astype(int)

# === China-related terms ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan", "tibet", "xinjiang",
    "one china", " prc ", " pla ", " ccp ", "belt and road", "silk road", 
    "spy balloon", "balloon incident", 
]
china_pronouns = {"they", "them", "their", "theirs"}

# === Precompile keyword regex ===
china_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k.lower()) for k in china_keywords) + r")\b"
)

# %%
# === Load spaCy + coreferee ===
print("Loading spaCy coreference model (this may take a moment)...")
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

# %%
# === Filter out filler / noise ===
def is_noise_sentence(text):
    text = text.strip().lower()
    return (
        text.startswith("(applause") or
        len(text) < 10 or
        re.match(r"^\(.*\)$", text)
    )

# %%
# === Extract China-related sentences with coref + precise matching ===
def extract_china_sentences_with_coref(speech):
    doc = nlp(speech)
    sents = list(doc.sents)
    china_related_sent_ids = set()

    # 1. Direct keyword match
    for i, sent in enumerate(sents):
        if china_pattern.search(sent.text.lower()):
            china_related_sent_ids.add(i)

    # 2. Coreference chains
    for chain in doc._.coref_chains:
        mentions = list(chain)
        chain_token_idxs = [idx for m in mentions for idx in m.token_indexes]

        chain_contains_china = any(
            china_pattern.search(doc[idx].text.lower()) for idx in chain_token_idxs
        )

        if not chain_contains_china:
            continue

        for m in mentions:
            if not m.token_indexes:
                continue
            tokens = [doc[idx] for idx in m.token_indexes]
            if any(t.text.lower() in china_pronouns or china_pattern.search(t.text.lower()) for t in tokens):
                sent = tokens[0].sent
                sent_id = next((i for i, s in enumerate(sents) if s.start == sent.start), None)
                if sent_id is not None:
                    china_related_sent_ids.add(sent_id)

    return [
        sents[i].text.strip()
        for i in sorted(china_related_sent_ids)
        if i < len(sents) and not is_noise_sentence(sents[i].text)
    ]

# %%
# === Load Cardiff NLP model ===
print("Loading CardiffNLP RoBERTa sentiment model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def compute_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    return scores[2] - scores[0]  # Compound-style sentiment: pos - neg

# %%
# === Analyze dataset ===
results = []
sentence_records = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    speech = row["text"]
    china_sents = extract_china_sentences_with_coref(speech)

    if china_sents:
        sentiment_scores = [compute_sentiment(s) for s in china_sents]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        mention_count = len(china_pattern.findall(" ".join(china_sents).lower()))

        results.append({
            "president": row.get("president", ""),
            "year": row.get("year", ""),
            "party": row.get("party", ""),
            "china_sentiment": avg_sentiment,
            "china_mention_count": mention_count
        })

        for sent, score in zip(china_sents, sentiment_scores):
            sentence_records.append({
                "president": row.get("president", ""),
                "year": row.get("year", ""),
                "party": row.get("party", ""),
                "sentence": sent,
                "sentiment": score
            })

# %%
# === Save outputs ===
china_df = pd.DataFrame(results)
china_df = china_df.groupby(["president", "year", "party"], as_index=False).agg({
    "china_sentiment": "mean",
    "china_mention_count": "sum"
})
china_df.to_csv(summary_csv, index=False)

sentence_df = pd.DataFrame(sentence_records)
sentence_df.to_csv(detailed_csv, index=False)

print("✅ Done! Saved:")
print(f"- Summary: {summary_csv}")
print(f"- Detailed: {detailed_csv}")
if not invalid_dates.empty:
    print(f"- Skipped rows with invalid dates: {skipped_log_csv}")
