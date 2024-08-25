import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Set up visualizations
sns.set(style="whitegrid")

# Add the project root directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

# Function Definitions
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def summary_statistics(df):
    return df.describe()

def check_missing_values(df):
    return df.isnull().sum()

def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def check_negative_values(df):
    ghi_negative = df[df['GHI'] < 0]
    dni_negative = df[df['DNI'] < 0]
    dhi_negative = df[df['DHI'] < 0]
    return ghi_negative, dni_negative, dhi_negative

def replace_negative_values(df):
    df['GHI'] = df['GHI'].apply(lambda x: max(x, 0))
    df['DNI'] = df['DNI'].apply(lambda x: max(x, 0))
    df['DHI'] = df['DHI'].apply(lambda x: max(x, 0))
    return df

def check_outliers(df, columns):
    return df[columns].describe()

def plot_boxplot(df, column):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column], ax=ax)
    st.pyplot(fig)

def plot_time_series_ghi(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    ax.plot(df['Timestamp'], df['GHI'], label='GHI', color='blue')
    ax.set_xlabel('Time')
    ax.set_ylabel('GHI (W/m²)')
    ax.set_title('Time Series of Global Horizontal Irradiance (GHI)')
    ax.legend()
    st.pyplot(fig)

def plot_time_series_multiple(df, columns):
    fig, ax = plt.subplots(figsize=(14, 7))
    for column in columns:
        ax.plot(df.index, df[column], label=column)
    ax.set_xlabel('Time')
    ax.set_ylabel('Measurements')
    ax.set_title('Time Series Analysis of Solar Radiation and Temperature')
    ax.legend()
    st.pyplot(fig)

def calculate_cleaning_effect(df):
    return df.groupby('Cleaning')[['ModA', 'ModB']].mean()

def plot_cleaning_impact(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    df[df['Cleaning'] == 1][['ModA', 'ModB']].plot(kind='line', ax=ax, title='Impact of Cleaning on Sensor Readings')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sensor Readings')
    st.pyplot(fig)

def plot_correlation_matrix(df, figsize=(16, 10), cmap='coolwarm'):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

def plot_correlation_heatmap(df, columns):
    corr_matrix = df[columns].corr()
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

def plot_pairplot(df, columns):
    fig = plt.figure(figsize=(12, 12))
    sns.pairplot(df[columns])
    st.pyplot(fig)

def plot_scatter_matrix(df, columns):
    fig, ax = plt.subplots(figsize=(12, 12))
    pd.plotting.scatter_matrix(df[columns], ax=ax)
    st.pyplot(fig)

def plot_polar_wind_analysis(df):
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    ax.scatter(df['WD'] * np.pi / 180, df['WS'], c=df['WSgust'], cmap='viridis', alpha=0.75)
    ax.set_title('Wind Speed and Direction')
    st.pyplot(fig)

def plot_temperature_vs_humidity(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='RH', y='Tamb', hue='GHI', ax=ax)
    ax.set_title('Temperature vs. Relative Humidity')
    st.pyplot(fig)

def plot_histograms(df, columns):
    fig, ax = plt.subplots(figsize=(15, 10))
    df[columns].hist(bins=30, ax=ax)
    plt.suptitle('Histograms of Key Variables')
    st.pyplot(fig)

def calculate_zscores(df):
    df['GHI_zscore'] = zscore(df['GHI'])
    df['DNI_zscore'] = zscore(df['DNI'])
    df['DHI_zscore'] = zscore(df['DHI'])
    return df

def plot_bubble_chart(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['GHI'], df['Tamb'], s=df['WS']*10, c=df['RH'], cmap='coolwarm', alpha=0.6)
    ax.set_xlabel('GHI (W/m²)')
    ax.set_ylabel('Ambient Temperature (°C)')
    ax.set_title('Bubble Chart: GHI vs. Tamb vs. WS (Bubble Size: RH)')
    fig.colorbar(scatter, ax=ax, label='Relative Humidity (%)')
    st.pyplot(fig)

def plot_ghi_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['GHI'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Global Horizontal Irradiance (GHI)')
    ax.set_xlabel('GHI (W/m²)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def plot_ghi_vs_dni(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='GHI', y='DNI', data=df, ax=ax)
    ax.set_title('GHI vs. DNI')
    ax.set_xlabel('GHI (W/m²)')
    ax.set_ylabel('DNI (W/m²)')
    st.pyplot(fig)

# Streamlit Dashboard
st.title('Solar Farm Data Analysis Dashboard')

# File selection
st.sidebar.header('Select Dataset')
option = st.sidebar.selectbox('Choose a dataset', ('Benin', 'Sierra Leone', 'Togo'))

file_paths = {
    'Benin': 'data/benin-malanville.csv',
    'Sierra Leone': 'data/sierraleone-bumbuna.csv',
    'Togo': 'data/togo-dapaong_qc.csv'
}

file_path = file_paths[option]
df = load_data(file_path)

st.write(f'### Data Preview: {option}')
st.write(df.head())

st.write('### Summary Statistics')
st.write(summary_statistics(df))

st.write('### Missing Values')
st.write(check_missing_values(df))

st.write('### Drop Columns')
df = drop_columns(df, ['Comments'])
st.write(df.head())

# st.write('### Negative Values Check')
# ghi_neg, dni_neg, dhi_neg = check_negative_values(df)
# st.write(f'Negative GHI Values:\n{ghi_neg}')
# st.write(f'Negative DNI Values:\n{dni_neg}')
# st.write(f'Negative DHI Values:\n{dhi_neg}')

df = replace_negative_values(df)

st.write('### Outliers Check')
st.write(check_outliers(df, ['ModA', 'ModB', 'WS', 'WSgust']))

st.write('### Boxplots')
plot_boxplot(df, 'ModA')
plot_boxplot(df, 'ModB')
plot_boxplot(df, 'WS')
plot_boxplot(df, 'WSgust')

st.write('### Time Series Analysis')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
plot_time_series_multiple(df, ['GHI', 'DNI', 'DHI', 'Tamb'])

st.write('### Impact of Cleaning')
st.write(calculate_cleaning_effect(df))
plot_cleaning_impact(df)

st.write('### Correlation Matrix')
plot_correlation_matrix(df)

st.write('### Correlation Heatmap')
plot_correlation_heatmap(df, ['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'WS', 'WSgust', 'WD'])

# st.write('### Pairplot')
# plot_pairplot(df, ['GHI', 'DNI', 'DHI', 'TModA', 'TModB'])

st.write('### Scatter Matrix')
plot_scatter_matrix(df, ['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'WD'])

st.write('### Polar Wind Analysis')
plot_polar_wind_analysis(df)

st.write('### Temperature vs. Humidity')
plot_temperature_vs_humidity(df)

st.write('### Histograms')
plot_histograms(df, ['GHI', 'DNI', 'DHI', 'Tamb'])

st.write('### Bubble Chart')
plot_bubble_chart(df)

st.write('### GHI Distribution')
plot_ghi_distribution(df)

st.write('### GHI vs. DNI')
plot_ghi_vs_dni(df)
