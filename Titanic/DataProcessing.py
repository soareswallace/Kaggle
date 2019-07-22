import pandas as pd


def load_data_from_csv(file_name):
    training_data = pd.read_csv(file_name)
    return training_data

