
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data ---------

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-03-25/report_words_clean.csv')

# clean data --------

# Define keywords related to data science
ds_keywords = ["data", "website", "cloud", "analysis", "software", "performance", "technology", "processing"]

# Filter the data to include only the words you're interested in
ds_data = report_words_clean[report_words_clean['word'].isin(ds_keywords)]

# Aggregate the word counts by year (since each word appears multiple times in the same year)
word_counts_by_year = ds_data.groupby(['year', 'word']).size().unstack(fill_value=0)

# Gaussian smoothing function
def gaussian_smooth(x, y, grid, sd):
    weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
    weights = weights / weights.sum(0)
    smoothed = (weights * y).sum(1)
    return smoothed

# Create a grid for the years (adjust the range and number of points as needed)
grid = np.linspace(2004, 2024, num=500)  # Creating a denser grid for smoother transition

# Prepare an empty DataFrame to store smoothed values
smoothed_df = pd.DataFrame(index=grid)

# Apply Gaussian smoothing for each word (column) in word_counts_by_year
for word in word_counts_by_year.columns:
    smoothed_df[word] = gaussian_smooth(word_counts_by_year.index.values, word_counts_by_year[word].values, grid, sd=0.55)  # Less smoothing here


smoothed_df = smoothed_df[smoothed_df.columns[::-1]]  # Reverse column order


# Plotting the smoothed data using stackplot
fig, ax = plt.subplots(figsize=(10, 7))



col = ['#a33a3a', '#ff8c86', '#ffaba4', '#ffcac3', '#acd6ec', '#8db7cc','#6F99AD', '#2c5769']

# col = ['#2c5769', '#6F99AD', '#8db7cc', '#acd6ec', '#ffcac3', '#ffaba4','#ff8c86', '#a33a3a']

# Create the stream plot (stackplot)
ax.stackplot(smoothed_df.index, smoothed_df.T, labels=smoothed_df.columns, alpha=0.7, colors=col, baseline="sym")

# Add labels and title
ax.set_title("Data Science Related Word Usage Over Time (Smoothed)", fontsize=16)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Word Frequency", fontsize=12)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='right')  # Reverse legend order


# Display the plot
plt.show()
