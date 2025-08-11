import pandas as pd

from utils import to_datetime

with open("data/energy_dataset.csv") as file:
    en = pd.read_csv(file, sep=",")
    en.name = "energy"
with open("data/weather_features.csv") as file:
    wt = pd.read_csv(file, sep=",")
    wt.name = "weather"

en = to_datetime(en, "time")
wt = to_datetime(wt, "dt_iso")
