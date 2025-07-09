# %%
import pandas as pd
import spacy
import coreferee
from textblob import TextBlob
import nltk
from tqdm import tqdm
import re

# Required once per session
nltk.download("punkt")

# %%
# === Load SOTU data ===
df = pd.read_csv("full_sotu_data.csv")
df["text"] = df["text"].fillna("")
df["year"] = df["year"].astype(int)

# === China-related terms ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan"
]

# %%
# === Load spaCy w/ coreferee ===
print("Loading spaCy coreference model (this may take a moment)...")
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("coreferee")

# %%
# === Helper: Extract China-related sentences from speech ===
def extract_china_sentences_with_coref(speech):
    doc = nlp(speech)
    china_related_sent_ids = set()
    
    # 1. Get sentence boundaries
    sents = list(doc.sents)
    
    # 2. Identify mentions of China
    china_token_indices = set()
    for i, token in enumerate(doc):
        if token.text.lower() in china_keywords:
            china_token_indices.add(i)

    # 3. Coreference chains
    for chain in doc._.coref_chains:
        mentions = list(chain)
        china_chain = any(
            idx in china_token_indices for m in mentions for idx in m.token_indexes
        )
        if china_chain:
            for m in mentions:
                if not m.token_indexes:
                    continue
                token_idx = m.token_indexes[0]
                sent_start = doc[token_idx].sent.start
                sent_id = next((i for i, s in enumerate(sents) if s.start == sent_start), None)
                if sent_id is not None:
                    china_related_sent_ids.add(sent_id)

    # 4. Also add directly mentioned sentences
    for i, sent in enumerate(sents):
        if any(k in sent.text.lower() for k in china_keywords):
            china_related_sent_ids.add(i)

    # 5. Return all tagged sentences (with bounds check)
    china_sentences = [sents[i].text for i in sorted(china_related_sent_ids) if i < len(sents)]
    return china_sentences

# %%
# === Process all speeches ===
results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    speech = row["text"]
    china_sents = extract_china_sentences_with_coref(speech)
    
    if china_sents:
        sentiment_scores = [TextBlob(s).sentiment.polarity for s in china_sents]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        mention_count = sum(
            len(re.findall(r"\b" + re.escape(k) + r"\b", " ".join(china_sents).lower()))
            for k in china_keywords
        )
        results.append({
            "president": row["president"],
            "year": row["year"],
            "party": row["party"],
            "china_sentiment": avg_sentiment,
            "china_mention_count": mention_count
        })

# %%
# === Convert to DataFrame & Save ===
china_df = pd.DataFrame(results)
china_df = china_df.groupby(["president", "year", "party"], as_index=False).agg({
    "china_sentiment": "mean",
    "china_mention_count": "sum"
})

china_df.to_csv("china_sentiment_coref_summary.csv", index=False)

print("✅ Saved: china_sentiment_coref_summary.csv")
