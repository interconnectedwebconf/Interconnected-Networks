import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from learn_matrices_with_recons_loss import InterpretableTensorClassifier

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InterpretableTensorClassifier(K=32, cp_rank=4).to(device)
model.load_state_dict(torch.load('interpretable_tensor_model.pth'))

# Extract the learned matrices
S1 = model.S1.detach().cpu().numpy()
S2 = model.S2.detach().cpu().numpy()
S3 = model.S3.detach().cpu().numpy()
S4 = model.S4.detach().cpu().numpy()

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Learned Interlink Matrices', fontsize=16)

# Plot each matrix as a heatmap
matrices = [S1, S2, S3, S4]
titles = ['S1', 'S2', 'S3', 'S4']

for idx, (matrix, title) in enumerate(zip(matrices, titles)):
    row = idx // 2
    col = idx % 2
    
    # Create heatmap
    sns.heatmap(matrix, 
                ax=axes[row, col],
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.2f',
                square=True)
    
    axes[row, col].set_title(f'{title} Matrix')
    axes[row, col].set_xlabel('Column')
    axes[row, col].set_ylabel('Row')

# Adjust layout and save
plt.tight_layout()
plt.savefig('learned_matrices_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics about the matrices
for title, matrix in zip(titles, matrices):
    print(f"\n{title} Matrix Statistics:")
    print(f"Mean: {np.mean(matrix):.4f}")
    print(f"Std: {np.std(matrix):.4f}")
    print(f"Min: {np.min(matrix):.4f}")
    print(f"Max: {np.max(matrix):.4f}")
