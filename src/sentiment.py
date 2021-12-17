from transformers import pipeline
import pandas as pd


def predict_sentiment():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    messages_df = pd.read_pickle("../pickle/parse.pkl")

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
    messages_df.to_pickle("../pickle/sentiment.pkl")


if __name__ == "__main__":
    predict_sentiment()
