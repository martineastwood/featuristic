import pandas as pd
import src as ft

df = pd.read_csv("examples/data/weather/Weather Training Data.csv")

schema = ft.featurize(df)

print(schema)
