import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('instance_stats_recons_loss.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Processing time vs Contact Network Size
sns.scatterplot(data=df, x='n_contact', y='elapsed_time', ax=ax1)
ax1.set_xlabel('Contact Network Size')
ax1.set_ylabel('Processing Time (seconds)')
ax1.set_title('Processing Time vs Contact Network Size')

# Plot 2: Processing time vs Communication Network Size
sns.scatterplot(data=df, x='n_comm', y='elapsed_time', ax=ax2)
ax2.set_xlabel('Communication Network Size')
ax2.set_ylabel('Processing Time (seconds)')
ax2.set_title('Processing Time vs Communication Network Size')

# Add a trend line to both plots
sns.regplot(data=df, x='n_contact', y='elapsed_time', ax=ax1, scatter=False, color='red')
sns.regplot(data=df, x='n_comm', y='elapsed_time', ax=ax2, scatter=False, color='red')

# Adjust layout and save
plt.tight_layout()
plt.savefig('processing_time_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics
print("\nProcessing Time Statistics:")
print(f"Average processing time: {df['elapsed_time'].mean():.2f} seconds")
print(f"Min processing time: {df['elapsed_time'].min():.2f} seconds")
print(f"Max processing time: {df['elapsed_time'].max():.2f} seconds")
print(f"Total processing time: {df['elapsed_time'].sum():.2f} seconds")

print("\nGraph Size Statistics:")
print("\nContact Network:")
print(f"Average size: {df['n_contact'].mean():.2f}")
print(f"Min size: {df['n_contact'].min()}")
print(f"Max size: {df['n_contact'].max()}")

print("\nCommunication Network:")
print(f"Average size: {df['n_comm'].mean():.2f}")
print(f"Min size: {df['n_comm'].min()}")
print(f"Max size: {df['n_comm'].max()}")