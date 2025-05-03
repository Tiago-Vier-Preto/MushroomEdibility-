import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data