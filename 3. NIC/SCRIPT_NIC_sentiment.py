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

nltk.download("punkt")

# %%
# === File Paths ===
folder = "3. NIC"
input_csv = os.path.join(folder, "NICtexts_updated.csv")
summary_csv = os.path.join(folder, "china_sentiment_summary.csv")
detailed_csv = os.path.join(folder, "china_sentences_detailed.csv")
skipped_log_csv = os.path.join(folder, "china_sentiment_skipped.csv")

# === Load data ===
df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()  # Clean whitespace in column headers
df["text"] = df["Full Text"].fillna("")

# === Parse and Normalize Years ===
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
invalid_dates = df[df["Year"].isna()]
if not invalid_dates.empty:
    print(f"⚠️ Skipping {len(invalid_dates)} rows with invalid years.")
    invalid_dates.to_csv(skipped_log_csv, index=False)

df = df[df["Year"].notna()]
df["Year"] = df["Year"].astype(int)

# === China-related terms ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan",
    "tibet", "xinjiang", "one china", "prc", "pla", "ccp",
    "belt and road", "silk road", "chinese economy", "trade with china",
    "chinese imports", "chinese exports", "tariffs on china",
    "south china sea", "military buildup", "chinese navy", "cyberattacks from china",
    "spy balloon", "balloon incident",
    "communist party", "authoritarian regime", "human rights in china",
    "democracy in hong kong"
]
china_pronouns = {"they", "them", "their", "theirs"}

china_pattern = re.compile(
    r"\b(" + "|".join(re.escape(k.lower()) for k in china_keywords) + r")\b"
)

# %%
# === Load spaCy + coreferee ===
print("🔁 Loading spaCy + coreferee...")
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

# %%
def is_noise_sentence(text):
    text = text.strip().lower()
    return (
        text.startswith("(applause") or
        len(text) < 10 or
        re.match(r"^\(.*\)$", text)
    )

# %%
def extract_china_sentences_with_coref(speech):
    doc = nlp(speech)
    sents = list(doc.sents)
    china_related_sent_ids = set()

    for i, sent in enumerate(sents):
        if china_pattern.search(sent.text.lower()):
            china_related_sent_ids.add(i)

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
print("🔁 Loading CardiffNLP sentiment model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def compute_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    return scores[2] - scores[0]  # pos - neg

# %%
# === Analyze dataset ===
results = []
sentence_records = []
skipped_rows = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    speech = str(row.get("text", "")).strip()

    # Skip empty or too short texts
    if not speech or len(speech) < 1000:
        skipped_rows.append(row)
        continue

    china_sents = extract_china_sentences_with_coref(speech)

    if china_sents:
        sentiment_scores = [compute_sentiment(s) for s in china_sents]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        mention_count = len(china_pattern.findall(" ".join(china_sents).lower()))

        results.append({
            "President": row.get("President", ""),
            "Year": row.get("Year", ""),
            "Party": row.get("Party", ""),
            "Role": row.get("Role", ""),
            "Intelligence Leader": row.get("Intelligence Leader", ""),
            "Publication Title": row.get("Publication Title", ""),
            "China Sentiment": avg_sentiment,
            "China Mention Count": mention_count
        })

        for sent, score in zip(china_sents, sentiment_scores):
            sentence_records.append({
                "President": row.get("President", ""),
                "Year": row.get("Year", ""),
                "Party": row.get("Party", ""),
                "Role": row.get("Role", ""),
                "Intelligence Leader": row.get("Intelligence Leader", ""),
                "Publication Title": row.get("Publication Title", ""),
                "Sentence": sent,
                "Sentiment": score
            })
    else:
        skipped_rows.append(row)

# %%
# === Save Outputs ===
summary_df = pd.DataFrame(results)
summary_df = summary_df.groupby(
    ["President", "Year", "Party", "Role", "Intelligence Leader", "Publication Title"],
    as_index=False
).agg({
    "China Sentiment": "mean",
    "China Mention Count": "sum"
})
summary_df.to_csv(summary_csv, index=False)

sentence_df = pd.DataFrame(sentence_records)
sentence_df.to_csv(detailed_csv, index=False)

if skipped_rows:
    pd.DataFrame(skipped_rows).to_csv(skipped_log_csv, index=False)

print("\n✅ Done!")
print(f"- Summary: {summary_csv}")
print(f"- Detailed: {detailed_csv}")
if skipped_rows:
    print(f"- Skipped rows saved to: {skipped_log_csv}")
