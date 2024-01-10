import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from collections import deque
import gc
import pickle


pd.set_option("display.float_format", lambda x: "%.0f" % x)


stock_csv_path = r"data/stock/min_2018.csv"


df = pd.read_csv(stock_csv_path)

df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]


df["Date"] = pd.to_datetime(df["timestamp"], unit="s")

df.drop("timestamp", axis=1, inplace=True)

df.ffill(inplace=True)
# df = df.drop("Adj Close", axis=1)


def create_features(df, only_technical_indicators=False):
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

    if not only_technical_indicators:
        df["Day_of_week"] = df["Date"].dt.weekday

        df["Is_weekend"] = df["Day_of_week"].isin([5, 6])
        df["Is_weekend"] = df["Is_weekend"].astype(int)

        df["Day_sin"] = np.sin(df["Day_of_week"] * (2 * np.pi / 7))
        df["Day_cos"] = np.cos(df["Day_of_week"] * (2 * np.pi / 7))

        # Assuming you have a "Time" column in your DataFrame
        df["Hour"] = df["Date"].dt.hour
        df["Minute"] = df["Date"].dt.minute

        # Calculate the sine and cosine transformations for time of day
        df["Time_sin"] = np.sin(
            (df["Hour"] * 60 + df["Minute"]) * (2 * np.pi / (24 * 60))
        )
        df["Time_cos"] = np.cos(
            (df["Hour"] * 60 + df["Minute"]) * (2 * np.pi / (24 * 60))
        )

        df.drop(
            [
                "Hour",
                "Minute",
                "Day_of_week",
            ],
            axis=1,
            inplace=True,
        )

    df["Future"] = df["Close"].shift(-1) > df["Close"]
    df["Future"] = df["Future"].astype(int)

    df.dropna(inplace=True)

    return df


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

input2_columns = [c + "_60_STEP" for c in input1_columns]

input3_columns = [c + "_1440_STEP" for c in input1_columns]


input4_columns = [
    "Is_weekend",
    "Day_sin",
    "Day_cos",
    "Time_sin",
    "Time_cos",
]


def preprocess_df(sub_df, scalers: dict, val=False, balance=True):
    for col, scaler in scalers.items():
        column_data = sub_df[col].values.reshape(-1, 1)
        if not val:
            scaler = scaler.fit(column_data)
            sub_df[col] = scaler.transform(column_data)
        else:
            sub_df[col] = scaler.transform(column_data)

    sub_df.dropna(inplace=True)

    sequential_data = []
    prev_mins = deque(maxlen=60)
    prev_hours = deque(maxlen=60)
    prev_days = deque(maxlen=60)

    for i1, i2, i3, i4, o in zip(
        sub_df[input1_columns].values,
        sub_df[input1_columns].values,
        sub_df[input3_columns].values,
        sub_df[input4_columns].values,
        sub_df["Future_60_STEP"].values,
    ):  # iterate over the values
        prev_mins.append(i1)
        prev_hours.append(i2)
        prev_days.append(i3)
        if len(prev_mins) == 60:  # largest one
            sequential_data.append(
                [np.array(prev_mins), np.array(prev_hours), np.array(prev_days), i4, o]
            )

    if balance:
        buys = []
        sells = []

        for x1, x2, x3, x4, y in sequential_data:
            if y == 1:
                buys.append([x1, x2, x3, x4, y])
            else:
                sells.append([x1, x2, x3, x4, y])

        min_len = min(len(buys), len(sells))

        buys = buys[:min_len]
        sells = sells[:min_len]

        print("split buys and sells")
        print(min_len)

        sequential_data = buys + sells

    np.random.shuffle(sequential_data)

    X1, X2, X3, X4, Y = zip(*sequential_data)

    del sequential_data

    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    # Y = np.array(np.eye(2)[Y])  # binary_crossentropy
    Y = np.array(Y)  # catergorical_crossentropy

    return X1, X2, X3, X4, Y


def add_pct_change(df, percent_change_scale):
    for col in percent_change_scale:
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
    return df


def create_window_dfs(
    df, window_size, percent_change_scale, only_technical_indicators=False
):
    dfs = []
    for i in range(window_size):
        new_df = df.shift(-i).iloc[::window_size, :]
        new_features_df = add_pct_change(
            create_features(new_df, only_technical_indicators), percent_change_scale
        )
        dfs.append(new_features_df)
    return pd.concat(dfs, axis=0)


def create_inputs(df):
    drop_cols = [
        "Low",
        "High",
        "Open",
    ]

    percent_change_scale = ["Close", "MA10", "MA20", "MA30", "Volume"]

    minute_df = add_pct_change(create_features(df.copy()), percent_change_scale)
    minute_df.drop(
        drop_cols,
        axis=1,
        inplace=True,
    )

    hour_df = create_window_dfs(
        df.copy(), 60, percent_change_scale, only_technical_indicators=True
    )
    hour_df.drop(
        drop_cols,
        axis=1,
        inplace=True,
    )
    hour_df = hour_df.add_suffix("_60_STEP")

    day_df = create_window_dfs(
        df.copy(), 1440, percent_change_scale, only_technical_indicators=True
    )
    day_df.drop(
        drop_cols,
        axis=1,
        inplace=True,
    )
    day_df = day_df.add_suffix("_1440_STEP")

    # print(minute_df.head())
    # print(minute_df.columns)
    # print(hour_df.head())
    # print(hour_df.columns)
    # print(day_df.head())
    # print(day_df.columns)
    # print(len(minute_df))
    # print(len(hour_df))
    # print(len(day_df))
    # print(len(df))

    minute_df.set_index("Date", inplace=True)
    hour_df.set_index("Date_60_STEP", inplace=True)
    day_df.set_index("Date_1440_STEP", inplace=True)

    df = pd.concat(
        [minute_df, hour_df, day_df],
        axis=1,
        join="outer",
    )

    del minute_df
    del hour_df
    del day_df

    df.dropna(inplace=True)

    # print(df.head())

    times = sorted(df.index.values)
    last_5pct = sorted(df.index.values)[-int(0.05 * len(times))]

    train_df = df[(df.index < last_5pct)]
    test_df = df[(df.index >= last_5pct)]

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

    to_scale1 = [c + "_60_STEP" for c in to_scale]
    to_scale2 = [c + "_1440_STEP" for c in to_scale]

    to_scale = to_scale + to_scale1 + to_scale2

    percent_change_scale1 = [c + "_60_STEP" for c in percent_change_scale]
    percent_change_scale2 = [c + "_1440_STEP" for c in percent_change_scale]

    percent_change_scale = (
        percent_change_scale + percent_change_scale1 + percent_change_scale2
    )

    scalers = {}
    for i in to_scale:
        scalers[i] = StandardScaler()

    train_x1, train_x2, train_x3, train_x4, train_y = preprocess_df(train_df, scalers)
    test_x1, test_x2, test_x3, test_x4, test_y = preprocess_df(
        test_df, scalers, val=True
    )

    return (
        train_x1,
        train_x2,
        train_x3,
        train_x4,
        train_y,
        test_x1,
        test_x2,
        test_x3,
        test_x4,
        test_y,
    )


(
    train_x1,
    train_x2,
    train_x3,
    train_x4,
    train_y,
    test_x1,
    test_x2,
    test_x3,
    test_x4,
    test_y,
) = create_inputs(create_features(df))


print(
    "Train Shapes:",
    [arr.shape for arr in [train_x1, train_x2, train_x3, train_x4, train_y]],
)
print(
    "Test Shapes:", [arr.shape for arr in [test_x1, test_x2, test_x3, test_x4, test_y]]
)


del df
gc.collect()

all_arrays = [
    train_x1,
    train_x2,
    train_x3,
    train_x4,
    train_y,
    test_x1,
    test_x2,
    test_x3,
    test_x4,
    test_y,
]
pickle.dump(all_arrays, open("data.pkl", "bw+"))
