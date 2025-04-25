import pandas as pd
import numpy as np

alpaca = pd.read_parquet("hf://datasets/tatsu-lab/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")
df = pd.read_json("/home/mjliu1/476project/dev_data.json")
print(df.head(5))
