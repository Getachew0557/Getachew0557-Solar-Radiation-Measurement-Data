import pandas as pd
import os

def load_data(filename):
    """
    Load dataset from the data folder.
    
    Parameters:
        filename (str): The name of the file to load.
        
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    data_path = os.path.join('data', filename)
    df = pd.read_csv(data_path)
    return df
