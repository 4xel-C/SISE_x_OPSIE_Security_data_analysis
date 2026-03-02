import pandas as pd

from services import DataManager

manager = DataManager(filename="1h-attack-log.csv")
df = manager.df
df = df.reset_index()

print(df.info())
