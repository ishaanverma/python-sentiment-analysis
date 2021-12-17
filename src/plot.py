import pandas as pd
import plotly.graph_objects as go


def plot():
    messages_df = pd.read_pickle("../pickle/sentiment.pkl")
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
    fig.write_image("../images/avg_sentiment.png")

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
    fig.write_image("../images/daily_count.png")


if __name__ == "__main__":
    plot()
