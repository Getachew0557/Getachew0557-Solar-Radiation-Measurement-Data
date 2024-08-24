import pandas as pd

def clean_data(df):
    """
    Perform data cleaning operations such as handling missing values.
    
    Parameters:
        df (DataFrame): The input DataFrame to clean.
        
    Returns:
        DataFrame: Cleaned DataFrame.
    """
    # Drop rows with missing values (example)
    df_cleaned = df.dropna()
    
    # Convert timestamp to datetime format
    if 'Timestamp' in df.columns:
        df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'])
    
    return df_cleaned
