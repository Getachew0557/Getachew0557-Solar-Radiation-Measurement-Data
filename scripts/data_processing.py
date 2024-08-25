import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Set up visualizations
sns.set(style="whitegrid")

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def summary_statistics(df):
    return df.describe()

def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)
    return missing_values

def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def check_negative_values(df):
    ghi_negative = df[df['GHI'] < 0]
    dni_negative = df[df['DNI'] < 0]
    dhi_negative = df[df['DHI'] < 0]
    print("Negative GHI Values:\n", ghi_negative)
    print("Negative DNI Values:\n", dni_negative)
    print("Negative DHI Values:\n", dhi_negative)
    return ghi_negative, dni_negative, dhi_negative

def replace_negative_values(df):
    df['GHI'] = df['GHI'].apply(lambda x: max(x, 0))
    df['DNI'] = df['DNI'].apply(lambda x: max(x, 0))
    df['DHI'] = df['DHI'].apply(lambda x: max(x, 0))
    return df

def check_outliers(df, columns):
    outliers = df[columns].describe()
    print("Outliers:\n", outliers)
    return outliers

def plot_boxplot(df, column):
    sns.boxplot(x=df[column])
    plt.show()

def plot_time_series_ghi(df):
    """
    Plot the time series of Global Horizontal Irradiance (GHI).
    """
    plt.figure(figsize=(14, 7))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    plt.plot(df['Timestamp'], df['GHI'], label='GHI', color='blue')
    plt.xlabel('Time')
    plt.ylabel('GHI (W/m²)')
    plt.title('Time Series of Global Horizontal Irradiance (GHI)')
    plt.legend()
    plt.show()

def plot_time_series_multiple(df, columns):
    """
    Plot time series data for multiple columns.
    """
    plt.figure(figsize=(14, 7))
    for column in columns:
        plt.plot(df.index, df[column], label=column)
    plt.xlabel('Time')
    plt.ylabel('Measurements')
    plt.title('Time Series Analysis of Solar Radiation and Temperature')
    plt.legend()
    plt.show()

def calculate_cleaning_effect(df):
    cleaning_effect = df.groupby('Cleaning')[['ModA', 'ModB']].mean()
    print("Impact of Cleaning on Sensor Readings:\n", cleaning_effect)
    return cleaning_effect

def plot_cleaning_impact(df):
    """
    Plot the impact of cleaning on sensor readings.
    """
    plt.figure(figsize=(14, 7))
    df[df['Cleaning'] == 1][['ModA', 'ModB']].plot(kind='line', title='Impact of Cleaning on Sensor Readings')
    plt.xlabel('Time')
    plt.ylabel('Sensor Readings')
    plt.show()

def plot_correlation_matrix(df, figsize=(16, 10), cmap='coolwarm'):
    """
    Calculate and visualize the correlation matrix using a heatmap.
    """
    corr_matrix = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_correlation_heatmap(df, columns):
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(16, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_pairplot(df, columns):
    sns.pairplot(df[columns])
    plt.show()

def plot_scatter_matrix(df, columns):
    pd.plotting.scatter_matrix(df[columns], figsize=(12, 12))
    plt.show()

def plot_polar_wind_analysis(df):
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, polar=True)
    ax.scatter(df['WD'] * np.pi / 180, df['WS'], c=df['WSgust'], cmap='viridis', alpha=0.75)
    plt.title('Wind Speed and Direction')
    plt.show()

def plot_temperature_vs_humidity(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='RH', y='Tamb', hue='GHI')
    plt.title('Temperature vs. Relative Humidity')
    plt.show()

def plot_histograms(df, columns):
    df[columns].hist(bins=30, figsize=(15, 10))
    plt.suptitle('Histograms of Key Variables')
    plt.show()

def calculate_zscores(df):
    df['GHI_zscore'] = zscore(df['GHI'])
    df['DNI_zscore'] = zscore(df['DNI'])
    df['DHI_zscore'] = zscore(df['DHI'])
    return df

def plot_bubble_chart(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['GHI'], df['Tamb'], s=df['WS']*10, c=df['RH'], cmap='coolwarm', alpha=0.6)
    plt.xlabel('GHI (W/m²)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.title('Bubble Chart: GHI vs. Tamb vs. WS (Bubble Size: RH)')
    plt.colorbar(label='Relative Humidity (%)')
    plt.show()

def plot_ghi_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['GHI'], bins=30, kde=True)
    plt.title('Distribution of Global Horizontal Irradiance (GHI)')
    plt.xlabel('GHI (W/m²)')
    plt.ylabel('Frequency')
    plt.show()

def plot_ghi_vs_dni(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GHI', y='DNI', data=df)
    plt.title('GHI vs. DNI')
    plt.xlabel('GHI (W/m²)')
    plt.ylabel('DNI (W/m²)')
    plt.show()
