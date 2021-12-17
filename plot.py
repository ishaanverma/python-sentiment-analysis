import pandas as pd
import plotly.graph_objects as go


def plot():
    messages_df = pd.read_pickle("sentiment.pkl")
    messages_df = messages_df.set_index("date")
    sentiment_mean_df = messages_df.resample("D").mean()
    fig = go.Figure(
        [go.Scatter(y=sentiment_mean_df["labels"], x=sentiment_mean_df.index)])
    fig.show()

    sentiment_count_df = messages_df.resample("D").count()
    fig = go.Figure(
        [go.Scatter(y=sentiment_count_df["message"], x=sentiment_count_df.index)])
    fig.show()


if __name__ == "__main__":
    plot()
