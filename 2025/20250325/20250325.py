
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data ---------

report_words_clean = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-03-25/report_words_clean.csv')

# clean data --------

# Get unique words for each year
unique_words_per_year = report_words_clean.groupby('year')['word'].unique()

# Convert to a DataFrame
unique_words_df = unique_words_per_year.reset_index()
