import pandas as pd
from pathlib import Path

path = Path("../data/customer_churn.csv")

def load_data(data_path):
    return pd.read_csv(data_path)

customer_data = load_data(path)