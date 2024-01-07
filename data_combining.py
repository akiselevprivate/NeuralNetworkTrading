import pandas as pd

base_path = "data/stock/"
file_paths = [
    "min_2017.csv",
    "min_2018.csv",
    "min_2019.csv",
    "min_2020.csv",
    "min_2021.csv",
]

dfs = []


for file_path in file_paths:
    df = pd.read_csv(base_path + file_path)
    dfs.append(df)

# Concatenate the list of DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

print(df.tail())
