import json
import re
import string
import pickle
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def contains_english_characters(sentence: str) -> bool:
    pattern = f"^[a-zA-Z0-9\s’{string.punctuation}]*$"
    matched = re.findall(pattern, sentence)

    if len(matched) > 0:
        return True
    return False


def contains_keywords(sentence, keywords):
    for word in word_tokenize(sentence):
        word = word.lower()
        if word in keywords:
            return True
    return False


def parse_json_file(file_name: str):
    with open(file_name, "r") as file:
        data = json.load(file)

    messages = data["messages"]
    parsed_messages = []
    parsed_timestamps = []
    for message in tqdm(messages):
        # Message text is a list, then convert to a string
        if isinstance(message["text"], list):
            for j, message_slice in enumerate(message["text"]):
                if isinstance(message_slice, dict):
                    message["text"][j] = message["text"][j]["text"]
            message["text"] = "".join(message["text"])

        # Check if message text contains english characters
        if contains_english_characters(message["text"]):
            if contains_keywords(message["text"], ["shib", "doge"]):
                parsed_messages.append(message["text"])
                parsed_timestamps.append(message["date"])

    df = pd.DataFrame(list(zip(parsed_timestamps, parsed_messages)), columns=[
        "date", "message"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_pickle("parse.pkl")


if __name__ == "__main__":
    parse_json_file("result.json")
