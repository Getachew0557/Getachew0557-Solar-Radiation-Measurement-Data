import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_basic_info(df):
    """
    Display basic information about the DataFrame.
    """
    print("DataFrame Shape:", df.shape)
    print("DataFrame Info:")
    print(df.info())
    print("DataFrame Description:")
    print(df.describe())

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix of the DataFrame.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
