import pandas as pd

def remove_duplicates(data):
    """Remove duplicate rows from the DataFrame."""
    initial_shape = data.shape
    data = data.drop_duplicates()
    final_shape = data.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    return data

def remove_attributes_with_missing_values(data, threshold=0.5):
    """Remove columns with a high percentage of missing values."""
    missing_percentage = data.isnull().mean()
    columns_to_remove = missing_percentage[missing_percentage > threshold].index
    data = data.drop(columns=columns_to_remove)
    print(f"Removed columns with more than {threshold*100}% missing values: {list(columns_to_remove)}")
    return data

def remove_specific_columns(data, columns_to_remove):
    """Remove specific columns from the DataFrame."""
    data = data.drop(columns=columns_to_remove, errors='ignore')
    print(f"Removed specified columns: {list(columns_to_remove)}")
    return data

def replace_missing_values_with_most_frequent(data, columns):
    """Replace missing values in specified columns with the most frequent value."""
    for column in columns:
        most_frequent_value = data[column].mode(dropna=True)[0]
        data[column] = data[column].fillna(most_frequent_value)
        print(f"Replaced missing values in '{column}' with '{most_frequent_value}'.")
    return data

def replace_missing_values_with_unknown(data, columns):
    """Replace missing values in specified columns with 'unknown'."""
    for column in columns:
        data[column] = data[column].fillna('?')
        print(f"Replaced missing values in {column} with 'unknown'.")
    return data

def transform_categorical_to_numerical(data, categorical_columns):
    """Convert categorical columns to numerical using one-hot encoding."""
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True, dtype=int)
    print(f"Transformed categorical columns to numerical: {list(categorical_columns)}")
    return data


def basic_info(data):	
    print("Dataset Information:")
    print(data.info())

    print("\nBasic Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())

def pre_process_data(data):
    data = remove_duplicates(data)
    data = remove_attributes_with_missing_values(data, threshold=0.5)
    data = remove_specific_columns(data, ['has-ring']) 
    data = replace_missing_values_with_most_frequent(data, ['gill-spacing'])
    data = replace_missing_values_with_unknown(data, ['cap-surface', 'gill-attachment', 'ring-type'])
    data = transform_categorical_to_numerical(data, ['cap-shape',  'cap-surface', 'cap-color', 'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color', 'stem-color', 'ring-type', 'habitat', 'season'])	
    data = data.reset_index(drop=True)
    return data
