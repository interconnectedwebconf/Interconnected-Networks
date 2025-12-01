import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# Set the style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 12

# Read the CSV file
df = pd.read_csv('instance_stats_recons_loss.csv')

# Function to remove outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from elapsed_time
df_clean = remove_outliers(df, 'elapsed_time')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2))  # Reduced height from 3.5 to 2.5

# Set the color palette
colors = sns.color_palette("husl", 2)

# Plot 1: Processing time vs Contact Network Size
scatter1 = sns.scatterplot(data=df_clean, x='n_contact', y='elapsed_time', 
                          ax=ax1, color=colors[0], alpha=0.6, s=60)  # Reduced marker size
line1 = sns.regplot(data=df_clean, x='n_contact', y='elapsed_time', 
                    ax=ax1, scatter=False, color='red', line_kws={'linewidth': 1.5})  # Reduced line width

ax1.set_xlabel('Contact Network Size', fontsize=8)
ax1.set_ylabel('Processing Time (seconds)', fontsize=8)

# Plot 2: Processing time vs Communication Network Size
scatter2 = sns.scatterplot(data=df_clean, x='n_comm', y='elapsed_time', 
                          ax=ax2, color=colors[1], alpha=0.6, s=60)  # Reduced marker size
line2 = sns.regplot(data=df_clean, x='n_comm', y='elapsed_time', 
                    ax=ax2, scatter=False, color='red', line_kws={'linewidth': 1.5})  # Reduced line width

ax2.set_xlabel('Communication Network Size', fontsize=8)
ax2.set_ylabel('Processing Time (seconds)', fontsize=8)

# Customize grid and format y-axis
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Format y-axis to show two decimal places
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    
    # Set y-axis ticks to show more values
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(np.linspace(y_min, y_max, 6))  # Show 6 evenly spaced ticks
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=10)

# Add more space between subplots
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Save the figure
plt.savefig('processing_time_analysis.pdf', 
            dpi=600, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
plt.close()
