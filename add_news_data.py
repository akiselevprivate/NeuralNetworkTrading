import pandas as pd
from db.models import *
import datetime
import time

from transformers import pipeline

pipe = pipeline("text-classification", model="ProsusAI/finbert")


def get_date(date):
    selected_tweets = (
        Tweet.select()
        .where(
            (
                Tweet.timestamp
                >= datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
            )
            & (
                Tweet.timestamp
                < (
                    datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
                    + datetime.timedelta(days=1)
                )
            )
        )
        .order_by(Tweet.likes.desc())
        .limit(100)
    )
    texts = [t.text for t in selected_tweets]
    return texts


def analyse_tweets(texts):
    if len(texts) == 0:
        return 0
    predictions = pipe(texts, top_k=None)
    prediction_dicts = [
        {item["label"]: item["score"] for item in prediction}
        for prediction in predictions
    ]
    positive = sum([pred["positive"] for pred in prediction_dicts])
    negative = sum([pred["negative"] for pred in prediction_dicts])

    score = (positive - negative) / len(prediction_dicts)
    return score


df = pd.read_csv(r"data/combined/Bitcoin_scaled_news.csv")

df["Date"] = pd.to_datetime(df["Date"])


length = len(df)
times = []

for index, row in df.iterrows():
    if pd.isna(row["Score"]):
        t1 = time.time()
        score = analyse_tweets(get_date(row["Date"]))

        df.at[index, "score"] = score

        # Save the DataFrame after each iteration (optional)
        df.to_csv(r"data/combined/Bitcoin_scaled_news.csv", index=False)

        t = time.time() - t1
        times.append(t)

        eta = sum(times) / len(times) * (length - index + 1)

    else:
        eta = 0
        score = None

    print(f"{index+1}/{length}, score: {score}, eta: {eta/60/60:.1f}hrs")
