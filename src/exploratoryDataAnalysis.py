import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def basic_info(data):	
    print("Dataset Information:")
    print(data.info())

    print("\nBasic Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())

def feature_analysis_categorical(data, target_column):
    plt.figure(figsize=(10, 6))
    data[target_column].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(f'{target_column}')
    plt.ylabel('Frequency')
    plt.show()

def feature_analysis_numerical(data, target_column):
    min_val = data[target_column].min()
    max_val = data[target_column].max()

    num_bins = 30
    bin_size = (max_val - min_val) / num_bins

    plt.figure(figsize=(10, 6))
    bins = np.linspace(start=min_val, stop=max_val, num=num_bins + 1)  
    data[target_column].plot(kind='hist', bins=bins, edgecolor='black', align='mid')
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(f'{target_column} (Min: {min_val}, Max: {max_val}, Bin Size: {bin_size:.2f})')
    plt.ylabel('Frequency')
    plt.show()

    print(f"\nStatistics for {target_column}:")
    print(data[target_column].describe())

def feature_analysis(data, target_column):
    if data[target_column].dtype == 'object':
        feature_analysis_categorical(data, target_column)
    else:
        feature_analysis_numerical(data, target_column)

def full_feature_analysis(data):
    for column in data.columns:
        feature_analysis(data, column)