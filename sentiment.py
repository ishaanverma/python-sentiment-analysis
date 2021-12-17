from transformers import pipeline
import pandas as pd


def predict_sentiment():
    classifier = pipeline("sentiment-analysis")
    messages_df = pd.read_pickle("./test.pkl")

    messages = messages_df["message"].values.tolist()
    sentiments = classifier(messages, model="distilbert-base-uncased-finetuned-sst-2-english")

    sentiment_labels = list(
        map(lambda item: 1 if item["label"] == "POSITIVE" else 0, sentiments))
    sentiment_score = list(map(lambda item: item["score"], sentiments))

    messages_df = messages_df.assign(labels=sentiment_labels)
    messages_df = messages_df.assign(score=sentiment_score)
    messages_df.to_pickle("sentiment.pkl")


if __name__ == "__main__":
    predict_sentiment()