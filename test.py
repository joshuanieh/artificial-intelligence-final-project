from distutils.log import error
import pandas as pd
import numpy as np
df = pd.read_csv("./data/keelung.csv", encoding="Big5").iloc[:, 3:]
df = df.apply(pd.to_numeric, errors='coerce')
print(df["0"].dtypes)
print(type(df.iloc[0,0]))
print(df.mean(axis=1, numeric_only=True))