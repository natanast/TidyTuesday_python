
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load data ---------

report_words_clean = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-03-25/report_words_clean.csv')

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
fig, ax = plt.subplots(figsize=(12, 10))


col = ['#a33a3a', '#ff8c86', '#ffaba4', '#ffcac3', '#acd6ec', '#8db7cc','#6F99AD', '#2c5769']

# col = ['#2c5769', '#6F99AD', '#8db7cc', '#acd6ec', '#ffcac3', '#ffaba4','#ff8c86', '#a33a3a']

# Create the stream plot (stackplot)
ax.stackplot(smoothed_df.index, smoothed_df.T, labels=smoothed_df.columns, alpha=0.7, colors=col, baseline="sym", edgecolor='white', linewidth=0.5)

# Add labels and title
# ax.set_title("Data Science Related Word Usage Over Time", fontsize=16)

ax.set_xlabel("Year", fontsize = 6)
ax.set_ylabel("Word Frequency", fontsize = 6)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=5)  # Move legend outside3

ax.set_xticks([2005, 2010, 2015, 2020, 2025])  # Set specific years as ticks
ax.set_xticklabels([2005, 2010, 2015, 2020, 2025], fontsize = 5)  # Ensure labels are readable
ax.set_yticklabels(ax.get_yticks(), fontsize=5)


plt.title(
    "Data Science Terminology in Amazon Annual Reports", 
    fontsize = 9,
    pad = 35,
    x = 0.5
    )

# Add the subtitle for clarification
plt.text(x = 0.5, y = 1.1, 
        s = "Tracking the Evolution of Data Science Keywords Over Time (2005–2025)", 
        ha = 'center', 
        va = 'center', 
        fontsize = 7, 
        style = 'italic', 
        color = "#8C8380",
        transform = plt.gca().transAxes)

# Add a caption to the plot
plt.text(
    x = 0.98, y = -0.3,  # Adjust x, y to position the caption below 
    s = "Source:  Amazon's annual reports | Graphic: Natasa Anastasiadou", 
    ha = 'center', 
    va = 'center', 
    fontsize = 5, 
    style = 'italic', 
    color = "#8C8380",
    transform = plt.gca().transAxes
)



# Display the plot
plt.tight_layout()
plt.show()


plt.savefig("plot.png", dpi = 600, bbox_inches='tight')  # Save with high resolution



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Define a smoothing function
def smooth_series(series, sigma=2):
    return gaussian_filter1d(series, sigma=sigma)

# Create a smoothed DataFrame
smoothed_df = pd.DataFrame(index=word_counts_by_year.index)

# Apply smoothing to each column
for column in word_counts_by_year.columns:
    smoothed_df[column] = smooth_series(word_counts_by_year[column], sigma=0.5)

# Define colors
colors = ['#a33a3a', '#ff8c86', '#ffaba4', '#ffcac3', '#acd6ec', '#8db7cc','#6F99AD', '#2c5769']

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot stacked area chart with smoothed data
ax.stackplot(smoothed_df.index, smoothed_df.T, labels=smoothed_df.columns, colors=colors, alpha=0.8)

# Labels and title
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Word Frequency", fontsize=10)
ax.set_title("Data Science Terminology in Amazon Reports", fontsize=12, fontweight="bold")

# Set x-ticks
ax.set_xticks(range(2005, 2025, 2))

# Add a legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

# Show the plot
plt.tight_layout()
plt.show()
