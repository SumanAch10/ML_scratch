
import pandas as pd
from pathlib import Path


data_path = Path("../data/housing.csv")

def load_data(data_path):
    return pd.read_csv(data_path)

housing_data = load_data(data_path = data_path)