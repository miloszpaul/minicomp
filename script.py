import pandas as pd

def imports():
    data_train = pd.read_csv('data/train_values.csv')
    data_labels = pd.read_csv('data/train_labels.csv')
    data_test = pd.read_csv('data/test_values.csv')