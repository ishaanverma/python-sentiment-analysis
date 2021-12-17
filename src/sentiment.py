import os
from transformers import pipeline
import pandas as pd
from constants import PICKLE_DIR


def predict_sentiment():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    messages_df = pd.read_pickle(os.path.join(PICKLE_DIR, "parse.pkl"))

    messages = messages_df["message"].values.tolist()
    sentiments = classifier(messages)

    sentiment_labels = list(
        map(lambda item: item["label"], sentiments)
    )
    sentiment_values = list(
        map(lambda item: 1 if item["label"] == "POSITIVE" else -1, sentiments)
    )
    sentiment_score = list(map(lambda item: item["score"], sentiments))

    messages_df = messages_df.assign(labels=sentiment_labels)
    messages_df = messages_df.assign(values=sentiment_values)
    messages_df = messages_df.assign(score=sentiment_score)
    messages_df.to_pickle(os.path.join(PICKLE_DIR, "sentiment.pkl"))


if __name__ == "__main__":
    predict_sentiment()
