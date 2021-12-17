import os
import pandas as pd
import plotly.graph_objects as go
from src.utils.constants import PICKLE_DIR, IMAGE_DIR


def plot(messages_df):
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
    fig.update_layout(
        title="Average Sentiment Per Day",
        xaxis_title="Time",
        yaxis_title="Average Sentiment [-1, 1]"
    )
    # fig.show()
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
    fig.update_layout(
        title="Messages Per Day",
        xaxis_title="Time",
        yaxis_title="Message Count"
    )
    # fig.show()
    fig.write_image(os.path.join(IMAGE_DIR, "daily_count.png"))


if __name__ == "__main__":
    messages_df = pd.read_pickle(os.path.join(PICKLE_DIR, "sentiment.pkl"))
    plot(messages_df)
