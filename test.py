import pandas as pd
import numpy as np
df = pd.read_csv("./data/keelung.csv", encoding="Big5").iloc[:, 3:]

print(df)
print(type(df.iloc[0,0]))
print(df.mean(axis=1, skipna=True, numeric_only=True))