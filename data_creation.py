import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
import time
from collections import deque


pd.set_option("display.float_format", lambda x: "%.0f" % x)


stock_csv_path = r"data/stock/min_2018.csv"


FUTURE_PERIOD_PREDICT = 3
SEQ_LEN = 60
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

df = pd.read_csv(stock_csv_path)

df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]


df["Date"] = pd.to_datetime(df["timestamp"], unit="s")

df.ffill(inplace=True)
# df = df.drop("Adj Close", axis=1)


# Moving Averages
df["MA10"] = ta.trend.sma_indicator(df["Close"], window=10)
df["MA20"] = ta.trend.sma_indicator(df["Close"], window=20)
df["MA30"] = ta.trend.sma_indicator(df["Close"], window=30)

# MACD
df["MACD"] = ta.trend.macd_diff(
    df["Close"], window_slow=26, window_fast=12, window_sign=9
)
df["MACD_SIGNAL"] = ta.trend.macd_signal(
    df["Close"], window_slow=26, window_fast=12, window_sign=9
)

# RSI
df["RSI6"] = ta.momentum.rsi(df["Close"], window=6)
df["RSI12"] = ta.momentum.rsi(df["Close"], window=12)
df["RSI24"] = ta.momentum.rsi(df["Close"], window=24)

# Money Flow Index (MFI)
df["MFI"] = ta.volume.money_flow_index(
    df["High"], df["Low"], df["Close"], df["Volume"], window=14
)


df["Day_of_week"] = df["Date"].dt.weekday

df["Is_weekend"] = df["Day_of_week"].isin([5, 6])
df["Is_weekend"] = df["Is_weekend"].astype(int)

df["Day_sin"] = np.sin(df["Day_of_week"] * (2 * np.pi / 7))
df["Day_cos"] = np.cos(df["Day_of_week"] * (2 * np.pi / 7))


# Assuming you have a "Time" column in your DataFrame
df["Hour"] = df["Date"].dt.hour
df["Minute"] = df["Date"].dt.minute

# Calculate the sine and cosine transformations for time of day
df["Time_sin"] = np.sin((df["Hour"] * 60 + df["Minute"]) * (2 * np.pi / (24 * 60)))
df["Time_cos"] = np.cos((df["Hour"] * 60 + df["Minute"]) * (2 * np.pi / (24 * 60)))


df["Future"] = df["Close"].shift(-3) > df["Close"]
df["Future"] = df["Future"].astype(int)

df.drop(
    [
        "Hour",
        "Minute",
        "Day_of_week",
    ],
    axis=1,
    inplace=True,
)

df.dropna(inplace=True)


# DATA CREATION END

input1_columns = [
    "Close",
    "MA10",
    "MA20",
    "MA30",
    "Volume",
    "MA10",
    "MA20",
    "MA30",
    "RSI6",
    "RSI12",
    "RSI24",
    "MFI",
    "MACD",
    "MACD_SIGNAL",
]

input3_columns = [
    "Is_weekend",
    "Day_sin",
    "Day_cos",
    "Time_sin",
    "Time_cos",
]


def preprocess_df(sub_df, pct_change, scalers: dict, val=False, balance=True):
    for col, scaler in scalers.items():
        if col in pct_change:
            sub_df[col] = sub_df[col].pct_change()
        sub_df.dropna(inplace=True)
        column_data = sub_df[col].values.reshape(-1, 1)
        if not val:
            scaler = scaler.fit(column_data)
            sub_df[col] = scaler.transform(column_data)
        else:
            sub_df[col] = scaler.transform(column_data)

    sub_df.dropna(inplace=True)

    # print(sub_df.tail())

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(
        maxlen=SEQ_LEN
    )  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i1, i3, o in zip(
        sub_df[input1_columns].values,
        sub_df[input3_columns].values,
        sub_df["Future"].values,
    ):  # iterate over the values
        prev_days.append(i1)  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append(
                [np.array(prev_days), i3, o]
            )  # append those bad boys!

    if balance:
        buys = []
        sells = []

        for x, x3, y in sequential_data:
            if y == 1:
                buys.append([x, x3, y])
            else:
                sells.append([x, x3, y])

        min_len = min(len(buys), len(sells))

        buys = buys[:min_len]
        sells = sells[:min_len]

        sequential_data = buys + sells

    np.random.shuffle(sequential_data)

    X1 = [i[0] for i in sequential_data]
    X3 = [i[1] for i in sequential_data]  # change the index
    Y = [i[2] for i in sequential_data]

    X1 = np.array(X1)
    X3 = np.array(X3)
    # Y = np.array(np.eye(2)[Y])  # binary_crossentropy
    Y = np.array(Y)  # catergorical_crossentropy

    return X1, X3, Y


df.drop(
    [
        "Low",
        "High",
        "Open",
    ],
    axis=1,
    inplace=True,
)

times = sorted(df.index.values)
last_5pct = sorted(df.index.values)[-int(0.05 * len(times))]

train_df = df[(df.index < last_5pct)]
test_df = df[(df.index >= last_5pct)]


percent_change_scale = ["Close", "MA10", "MA20", "MA30", "Volume"]

to_scale = [
    "MA10",
    "MA20",
    "MA30",
    "RSI6",
    "RSI12",
    "RSI24",
    "MFI",
    "MACD",
    "MACD_SIGNAL",
] + percent_change_scale


scalers = {}
for i in to_scale:
    scalers[i] = StandardScaler()


train_x1, train_x3, train_y = preprocess_df(train_df, percent_change_scale, scalers)
test_x1, test_x3, test_y = preprocess_df(
    test_df, percent_change_scale, scalers, val=True
)


print(train_x1.shape, train_x3.shape, train_y.shape)
print(test_x1.shape, test_x3.shape, test_y.shape)
