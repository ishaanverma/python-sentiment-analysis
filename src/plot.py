import os
import pandas as pd
import plotly.graph_objects as go
from constants import PICKLE_DIR, IMAGE_DIR


def plot():
    messages_df = pd.read_pickle(os.path.join(PICKLE_DIR, "sentiment.pkl"))
    messages_df = messages_df.set_index("date")
    sentiment_mean_df = messages_df.resample("D").mean()
    fig = go.Figure(
        [
            go.Scatter(
                y=sentiment_mean_df["values"],
                x=sentiment_mean_df.index)
        ],
        layout_yaxis_range=[-1, 1]
    )
    fig.show()
    fig.write_image(os.path.join(IMAGE_DIR, "avg_sentiment.png"))

    sentiment_count_df = messages_df.resample("D").count()
    fig = go.Figure(
        [
            go.Scatter(
                y=sentiment_count_df["message"],
                x=sentiment_count_df.index
            )
        ]
    )
    fig.show()
    fig.write_image(os.path.join(IMAGE_DIR, "daily_count.png"))


if __name__ == "__main__":
    plot()
