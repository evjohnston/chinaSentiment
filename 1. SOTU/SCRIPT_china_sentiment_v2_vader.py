# %%
import pandas as pd
import spacy
import coreferee
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import re

# Download necessary resources
nltk.download("punkt")
nltk.download("vader_lexicon")

# %%
# === Load data ===
df = pd.read_csv("CSV_full_sotu_data.csv")
df["text"] = df["text"].fillna("")
df["year"] = df["year"].astype(int)

# === China-related terms ===
china_keywords = [
    "china", "chinese", "beijing", "xi jinping", "hong kong", "taiwan",
    "tibet", "xinjiang", "one china", "prc", "pla", "ccp",
    "belt and road", "silk road", "chinese economy", "trade with china",
    "chinese imports", "chinese exports", "tariffs on china",
    "south china sea", "military buildup", "chinese navy", "cyberattacks from china",
    "spy balloon", "balloon incident",
    "communist party", "ccp", "authoritarian regime", "human rights in china",
    "democracy in hong kong"
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
# === Initialize VADER ===
sia = SentimentIntensityAnalyzer()

# %%
# === Analyze dataset ===
results = []
sentence_records = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    speech = row["text"]
    china_sents = extract_china_sentences_with_coref(speech)

    if china_sents:
        sentiment_scores = [sia.polarity_scores(s)["compound"] for s in china_sents]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        mention_count = len(china_pattern.findall(" ".join(china_sents).lower()))

        results.append({
            "president": row["president"],
            "year": row["year"],
            "party": row["party"],
            "china_sentiment": avg_sentiment,
            "china_mention_count": mention_count
        })

        for sent, score in zip(china_sents, sentiment_scores):
            sentence_records.append({
                "president": row["president"],
                "year": row["year"],
                "party": row["party"],
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
china_df.to_csv("CSV_china_sentiment_coref_summary.csv", index=False)

sentence_df = pd.DataFrame(sentence_records)
sentence_df.to_csv("CSV_china_sentences_coref_detailed.csv", index=False)

print("✅ Done! Saved:")
print("- CSV_china_sentiment_coref_summary.csv")
print("- CSV_china_sentences_coref_detailed.csv")
