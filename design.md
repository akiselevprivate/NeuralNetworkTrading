# Trend prediction using neural networks

# Model Data

Stock:

- Close Price
- Volume

News:

- avg setiment score
- news volume

Rething the daily score function to include neutral score

Technical indicators:

- MA10 moving avarage 10 days
- MA20 moving avarage 20 days
- MA30 moving avarage 30 days
- MACD (DIFF) The difference between EMA12
- MACD (DEA) 9-day exponential moving average
- MACD moving avarage convergance and divergence
- RSI6 6 day relative strength index
- RSI12 12 day relative strength index
- RSI24 24 day relative strength index
- MFI money flow index

# Model Design

## Time series data

Window size: 50

Steps:

- get future ready
- add technical indicators
- convert needed indicators to percent change
- normalise the indicators (0, 1)
- daily data -> price percentage change -> normalised (0, 1)

Layers:

1. LSTM 128
2. Dropout 0.2
3. LSTM 128
4. Dropout 0.1
5. LSTM 128
6. Dropout 0.2

## News data

Window size: 7

Steps:

- collect top 100 tweets sorted in ascending order by acount followers
- get the avarage sentiment value of the tweets positive and negative
- scale the sentiment (-1, 1)

Layers:

1. LSTM 32
2. Dropout 0.2
3. LSTM 32
4. Dropout 0.2

## Financial Indicators

Steps:

For each value have a current and previous value.

Values:

- Inflation
- Interest
- ?gold price

## Other Features

Use One-Hot encoding for days of the week (7 different features)

- Day of the week
- Time of day

## Concatenation layer

Layers:

1. Dense 128 relu
2. Dense 64 relu
3. Dense 32 relu
4. Dense 2 softmax

# Comments

for the trading bot itself use Volatility Measures

Stop-Loss and Take-Profit Orders: Setting predefined levels to limit losses or lock in profits.  
Position Sizing: Determining the size of each trade relative to the overall trading capital.

maybe add high/low features

how diverce the news sentiment is

# Results

stock (56%)

stock+features 63.6% not finished training, val acc 54.7%, i think its overfitting cuz only 128 lstm nodes

stock+features 57% not finished, val acc 56%, 265 lstm

stock+features 63% not finished, val acc 56%, 128 lstm, overfitting

stock+features 58% not finished, val acc 58%, 128 lstm, 20 epochs (11 is best)

stock+features 57% loss, val acc 55% stalled at the end 11 epochs, 128 nodes

# Input Data

- General Twitter news
- Influencial Twitter news
- General Reddit news
- General Twitter Crypto-currecy news
- General Reddit Crypto-currecy news
- Time series data
- Date, Time
- Interest, GDP, Inflation (previous and current)
