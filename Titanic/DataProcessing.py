import pandas as pd


def load_data_from_csv(file_name, drop_na):
    training_data = pd.read_csv(file_name)
    if drop_na:
        training_data = training_data.fillna(1)
    return training_data

