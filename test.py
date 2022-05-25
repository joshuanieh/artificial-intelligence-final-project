import pandas as pd
import numpy as np
df = pd.read_csv("./data/keelung.csv", encoding="Big5", skiprows=2, usecols=[i for i in range(3,27)])

# print(df)
# print(df.iloc[0,0])
# df.astype(dtype='float64', errors='ignore')
df = df.apply(pd.to_numeric, errors='coerce')
print(df)
print(df.mean(axis=1, skipna=True, numeric_only=True))