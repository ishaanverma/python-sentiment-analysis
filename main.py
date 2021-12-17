from src.parse import parse_json_file
from src.sentiment import predict_sentiment
from src.plot import plot


def main():
    # Parse input file
    print("PARSING INPUT FILE...")
    messages_df = parse_json_file("result.json")

    # Classify each message
    print("PREDICTING SENTENCE SENTIMENT...")
    messages_df = predict_sentiment(messages_df)

    # Plot Sentitment Data
    plot(messages_df)
    print("SUCCESS")


if __name__ == "__main__":
    main()
