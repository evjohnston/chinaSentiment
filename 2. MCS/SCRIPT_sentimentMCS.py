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

# NLTK sentence tokenizer
nltk.download("punkt")

# === File Paths ===
folder = "2. MCS"
input_csv = os.path.join(folder, "miller_center_speeches.csv")
summary_csv = os.path.join(folder, "china_sentiment_coref_summary.csv")
detailed_csv = os.path.join(folder, "china_sentences_coref_detailed.csv")
skipped_log_csv = os.path.join(folder, "skipped_rows_due_to_invalid_dates.csv")

# === Load data ===
df = pd.read_csv(input_csv)
df["text"] = df["transcript"].fillna("")

# === Normalize Dates ===
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
invalid_dates = df[df["date_parsed"].isna()]
if not invalid_dates.empty:
    print(f"⚠️ Skipping {len(invalid_dates)} rows with invalid or unparseable dates.")
    invalid_dates.to_csv(skipped_log_csv, index=False)

df = df[df["date_parsed"].notna()]
df["date"] = df["date_parsed"].dt.strftime("%Y-%m-%d")
df["year"] = df["date_parsed"].dt.year

# === President-to-Party Mapping ===
party_map = {
    "Abraham Lincoln": "Republican", "Ulysses S. Grant": "Republican", "Rutherford B. Hayes": "Republican",
    "James Garfield": "Republican", "Chester Arthur": "Republican", "Benjamin Harrison": "Republican",
    "William McKinley": "Republican", "Theodore Roosevelt": "Republican", "William Howard Taft": "Republican",
    "Warren Harding": "Republican", "Calvin Coolidge": "Republican", "Herbert Hoover": "Republican",
    "Dwight Eisenhower": "Republican", "Richard Nixon": "Republican", "Gerald Ford": "Republican",
    "Ronald Reagan": "Republican", "George Bush": "Republican", "George W. Bush": "Republican",
    "Donald Trump": "Republican",
    "Andrew Jackson": "Democrat", "Martin Van Buren": "Democrat", "James K. Polk": "Democrat",
    "Franklin Pierce": "Democrat", "James Buchanan": "Democrat", "Grover Cleveland": "Democrat",
    "Woodrow Wilson": "Democrat", "Franklin D. Roosevelt": "Democrat", "Harry Truman": "Democrat",
    "John F. Kennedy": "Democrat", "Lyndon Johnson": "Democrat", "Jimmy Carter": "Democrat",
    "Bill Clinton": "Democrat", "Barack Obama": "Democrat", "Joe Biden": "Democrat",
    "George Washington": "Federalist", "John Adams": "Federalist",
    "Thomas Jefferson": "Democratic-Republican", "James Madison": "Democratic-Republican",
    "James Monroe": "Democratic-Republican", "John Quincy Adams": "Democratic-Republican",
    "William Henry Harrison": "Whig", "John Tyler": "Whig", "Zachary Taylor": "Whig", "Millard Fillmore": "Whig",
    "Andrew Johnson": "Union"
}
df["party"] = df["president"].map(party_map).fillna("Unknown")

# === China keywords and coref targets ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan", "tibet", "xinjiang",
    "one china", " prc ", " pla ", " ccp ", "belt and road", "silk road", 
    "spy balloon", "balloon incident", 
]
china_pronouns = {"they", "them", "their", "theirs"}
china_pattern = re.compile(r"\b(" + "|".join(re.escape(k.lower()) for k in china_keywords) + r")\b")

# === Load spaCy + coreferee ===
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

# === Filter function for noise ===
def is_noise_sentence(text):
    text = text.strip().lower()
    return text.startswith("(applause") or len(text) < 10 or re.match(r"^\(.*\)$", text)

# === Extract China-related sentences with coref support ===
def extract_china_sentences_with_coref(speech):
    doc = nlp(speech)
    sents = list(doc.sents)
    china_related_sent_ids = set()

    for i, sent in enumerate(sents):
        if china_pattern.search(sent.text.lower()):
            china_related_sent_ids.add(i)

    for chain in doc._.coref_chains:
        mentions = list(chain)
        token_idxs = [idx for m in mentions for idx in m.token_indexes]
        if any(china_pattern.search(doc[idx].text.lower()) for idx in token_idxs):
            for m in mentions:
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

# === Load CardiffNLP model ===
print("Loading CardiffNLP RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def compute_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    return scores[2] - scores[0]

# === Run analysis ===
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
            "title": row.get("title", ""),
            "president": row.get("president", ""),
            "year": row.get("year", ""),
            "party": row.get("party", ""),
            "china_sentiment": avg_sentiment,
            "china_mention_count": mention_count
        })

        for sent, score in zip(china_sents, sentiment_scores):
            sentence_records.append({
                "title": row.get("title", ""),
                "president": row.get("president", ""),
                "year": row.get("year", ""),
                "party": row.get("party", ""),
                "sentence": sent,
                "sentiment": score
            })

# === Save outputs ===
pd.DataFrame(results).to_csv(summary_csv, index=False)
pd.DataFrame(sentence_records).to_csv(detailed_csv, index=False)

print("✅ Done! Saved:")
print(f"- Summary: {summary_csv}")
print(f"- Detailed: {detailed_csv}")
if not invalid_dates.empty:
    print(f"- Skipped rows with invalid dates: {skipped_log_csv}")
