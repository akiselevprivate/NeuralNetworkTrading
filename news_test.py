from db.old_models import WriteTweet, ReadTweet
import datetime
from playhouse.shortcuts import model_to_dict

# from transformers import pipeline
import time

# pipe = pipeline("text-classification", model="ProsusAI/finbert")


analyse_tweets_per_day = 100


def classify_dates(dates):
    tweet_dicts = []
    chunk_lengths = []

    empty_days = 0

    non_classified_dates = []
    non_classified_tweet_dicts = []

    classified_dates_score = []
    classified_dates = []

    t1 = time.time()

    for idx, date in enumerate(dates):
        selected_tweets = (
            ReadTweet.select()
            .where(
                (
                    ReadTweet.date
                    >= datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
                )
                & (
                    ReadTweet.date
                    < (
                        datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
                        + datetime.timedelta(days=1)
                    )
                )
            )
            .group_by(ReadTweet.user_name)
            .order_by(ReadTweet.user_followers.desc())
            .limit(analyse_tweets_per_day)
        )

        selected_existing_tweets = (
            WriteTweet.select()
            .where(
                (
                    WriteTweet.date
                    >= datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
                )
                & (
                    WriteTweet.date
                    < (
                        datetime.datetime(date.year, date.month, date.day, 0, 0, 0)
                        + datetime.timedelta(days=1)
                    )
                )
            )
            .group_by(WriteTweet.user_name)
            .order_by(WriteTweet.user_followers.desc())
            .limit(analyse_tweets_per_day)
        )

        new_tweet_dicts = [model_to_dict(t) for t in selected_tweets]
        existing_tweets = [model_to_dict(t) for t in selected_existing_tweets]

        print(len(new_tweet_dicts), len(existing_tweets))

        all_dates_classified = all([x["positive"] for x in existing_tweets])

        if len(existing_tweets) == 0 or len(new_tweet_dicts) == 0:
            classified_dates.append(date)
            classified_dates_score.append(0)
            continue

        if len(existing_tweets) == len(new_tweet_dicts) and all_dates_classified:
            print(True)
            positive = sum([pred["positive"] for pred in existing_tweets])
            negative = sum([pred["negative"] for pred in existing_tweets])
            score = (positive - negative) / len(existing_tweets)
            classified_dates_score.append(score)
            classified_dates.append(date)
            tweet_dicts += existing_tweets
        else:
            print(False)
            if len(new_tweet_dicts) == 0:
                empty_days += 1

            else:
                chunk_lengths.append(len(new_tweet_dicts))
                tweet_dicts += new_tweet_dicts
                non_classified_tweet_dicts.append(new_tweet_dicts)
                non_classified_dates.append(date)

        print(f"{int(idx + 1)}/{len(dates)}")

    print("empty days:", empty_days)
    print(f"{len(tweet_dicts)}/{len(dates)*analyse_tweets_per_day}")
    print("data time:", time.time() - t1)

    predictions = pipe([t["text"] for t in non_classified_tweet_dicts], top_k=None)

    prediction_dicts = [
        {item["label"]: item["score"] for item in prediction}
        for prediction in predictions
    ]

    non_classified_scores = []

    print("saving data")

    t2 = time.time()

    c = 0
    for i in chunk_lengths:
        local_prediction_dicts = prediction_dicts[c : c + i]
        local_tweet_dicts = non_classified_tweet_dicts[c : c + i]
        positive = sum([pred["positive"] for pred in local_prediction_dicts])
        negative = sum([pred["negative"] for pred in local_prediction_dicts])

        for t, pred in zip(local_tweet_dicts, local_prediction_dicts):
            del t["positive"]
            del t["neutral"]
            del t["negative"]
            try:
                WriteTweet.create(
                    **t,
                    positive=pred["positive"],
                    neutral=pred["neutral"],
                    negative=pred["negative"],
                )
            except:
                pass

        # if len(prediction_dicts) == 0:
        #     scores.append(0)
        # else:
        score = (positive - negative) / len(prediction_dicts)
        scores.append(score)

    scores = []

    for date in dates:
        if date in non_classified_dates:
            idx = non_classified_dates.index(date)
            score = non_classified_scores[idx]
            scores.append(score)
        elif date in classified_dates:
            idx = classified_dates.index(date)
            score = classified_dates_score[idx]
            scores.append(score)
        else:
            raise Exception("date not found in any of the dicts")

    print("data save time:", time.time() - t2)
    print("time taken:", time.time() - t1)

    return scores


scores = classify_dates([datetime.date(2021, 2, 10), datetime.date(2021, 2, 15)])
