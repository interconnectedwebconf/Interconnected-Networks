from util import *
from soft_clustering import *
import numpy as np
import random
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split

def reconstruct_matrix_from_svd(A, k):
    """Reconstructs a matrix from its top-k singular components."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(S)
    S_sqrt = np.sqrt(S)
    Z1 = U @ S_sqrt
    Z2 = S_sqrt @ Vt
    return Z1, Z2  # shape: K x K

def build_tensor_input(file_path, rank = 32):
    """
    Loads the super-adjacency matrix from a file and computes its SVD features.

    Args:
        file_path (str): Path to the file containing the super-adjacency
            matrix.
        top_k (int): Number of top singular values to retain. If None, keep all.
    Returns:
        np.ndarray: Sorted vector of top_k singular values (descending).
    """
    processed_instance = process_instance(file_path)
    A_ic = processed_instance['super_comm_adj']
    A_sir = processed_instance['super_contact_adj']
    A_inter = processed_instance['super_interlink_adj']
    label = processed_instance['label']
    contact_cost = processed_instance['contact_cost']
    comm_cost = processed_instance['comm_cost']
    gamma = processed_instance['gamma']
    behavioral_change_parameter = processed_instance['behavioral_change_parameter']
    
    Z1_ic, Z2_ic = reconstruct_matrix_from_svd(A_ic, rank)
    Z1_sir, Z2_sir = reconstruct_matrix_from_svd(A_sir, rank)
    Z1_inter, Z2_inter = reconstruct_matrix_from_svd(A_inter, rank)

    # # Apply SVD-based reconstruction to each matrix
    # A_ic_k = reconstruct_matrix_from_svd(A_ic, rank)
    # A_sir_k = reconstruct_matrix_from_svd(A_sir, rank)
    # A_inter_k = reconstruct_matrix_from_svd(A_inter, rank)
    
    

    # Stack into a tensor: shape [6, K, K]
    tensor_input = np.stack([Z1_ic, Z2_ic, Z1_sir, Z2_sir, Z1_inter, Z2_inter])

    # Include seed vectors as separate features (optional)
    seed_ic = processed_instance['super_comm_seed']  # shape [K]
    seed_sir = processed_instance['super_contact_seed']  # shape [K]
    
    seed_ic_conv = Z1_ic @ seed_ic
    seed_sir_conv = Z1_sir @ seed_sir

    return {
        'tensor': tensor_input,        # shape [6, K, K] 
        'seed_ic': seed_ic,            # shape [K]
        'seed_sir': seed_sir,           # shape [K]
        'seed_ic_conv': seed_ic_conv,  # shape [K]
        'seed_sir_conv': seed_sir_conv,  # shape [K]
        'label': label,                # label
        'contact_cost': contact_cost,  # scaler 
        'comm_cost': comm_cost,        # scaler 
        'gamma': gamma,                # scaler 
        'behavioral_change_parameter': behavioral_change_parameter  # scaler 
    }   


def cp_decompose_tensor(tensor, rank):
    """
    Perform CP decomposition on a single 3D tensor.
    
    Args:
        tensor: a 3D torch.Tensor of shape (3, K, K)
        rank: number of components for CP decomposition
    
    Returns:
        weights: vector of length `rank`
        factors: list of factor matrices [A, B, C]
    """
    # Convert to float (if needed)
    tensor = tensor.to(dtype=torch.float32)
    
    # Perform CP decomposition
    weights, factors = parafac(tensor, rank=rank, init='svd', normalize_factors=True)
    
    return weights, factors  # factors = [A, B, C] with shapes (3, R), (K, R), (K, R)

####################################################################
# classification model building
####################################################################

class TensorCPClassifier(nn.Module):
    def __init__(self, K, cp_rank, hidden_dim=64):
        super(TensorCPClassifier, self).__init__()
        self.K = K
        self.cp_rank = cp_rank
        
        # Seed vector + convolutional versions: [4*K]
        self.seed_fc = nn.Sequential(
            nn.Linear(4 * K, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Scalar input: [4]
        self.scalar_fc = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final classifier: CP embedding + seed features + scalar features
        self.classifier = nn.Sequential(
            nn.Linear(cp_rank * (6+2*K) + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # binary classification
        )

    def forward(self, tensor_input, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars):
        # tensor_input: [B, 6, K, K]
        B = tensor_input.shape[0]
        cp_feats = []

        for i in range(B):
            cp_result = parafac(tensor_input[i], rank=self.cp_rank, init='svd', n_iter_max=10)
            # Extract factor matrices and flatten
            factors = cp_result.factors  # List of 3 tensors (6, r), (K, r), (K, r)
            flatten = torch.cat([f.view(-1) for f in factors], dim=0)
            cp_feats.append(flatten)
        cp_feats = torch.stack(cp_feats, dim=0)
        
        # Ensure all inputs have the same dtype
        seed_ic = seed_ic.float()
        seed_sir = seed_sir.float()
        seed_ic_conv = seed_ic_conv.float()
        seed_sir_conv = seed_sir_conv.float()
        scalars = scalars.float()

        # Seed features
        seed_feat = torch.cat([seed_ic, seed_sir, seed_ic_conv, seed_sir_conv], dim=1)
        seed_out = self.seed_fc(seed_feat)
        seed_out = seed_out.float()

        # Scalar features
        scalar_out = self.scalar_fc(scalars)
        scalar_out = scalar_out.float()
        # Combine all
        combined = torch.cat([cp_feats, seed_out, scalar_out], dim=1)
        combined = combined.float()
        
        print(combined.dtype)
        return self.classifier(combined).squeeze()
    
class InterpretableTensorClassifier(nn.Module):
    def __init__(self, K, cp_rank, hidden=32):
        super().__init__()
        self.cp_rank = cp_rank

        # One MLP per CP rank component
        self.cp_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(cp_rank)
        ])

        # One MLP per seed vector
        self.seed_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(K, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(4)
        ])

        # Final classifier: input size = cp_rank + 4
        self.final = nn.Sequential(
            nn.Linear(cp_rank + 8, hidden),  # 8 = 4 seed scalars + 4 raw scalars
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, tensor_input, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars):
        # Ensure all tensors are float32
        tensor_input = tensor_input.float()
        seed_ic = seed_ic.float()
        seed_sir = seed_sir.float()
        seed_ic_conv = seed_ic_conv.float()
        seed_sir_conv = seed_sir_conv.float()
        scalars = scalars.float()

        B = tensor_input.size(0)
        output_list = []

        for i in range(B):
            cp_result = parafac(tensor_input[i], rank=self.cp_rank, init='svd', n_iter_max=10)
            factors = cp_result.factors  # A (6, r), B (K, r), C (K, r)

            # CP scalars
            scalars_cp = []
            for r in range(self.cp_rank):
                triple = torch.stack([
                    factors[0][:, r].mean(),
                    factors[1][:, r].mean(),
                    factors[2][:, r].mean()
                ])
                triple = triple.float()  # ensure triple is float32
                scalars_cp.append(self.cp_mlps[r](triple.unsqueeze(0)).squeeze())

            # Seed vector scalars
            seeds = [seed_ic[i], seed_sir[i], seed_ic_conv[i], seed_sir_conv[i]]
            scalars_seed = []
            for j in range(4):
                vector = seeds[j].float().unsqueeze(0)  # shape: [1, K]
                scalars_seed.append(self.seed_mlps[j](vector).squeeze())

            # Scalar features (already float32)
            scalars_raw = scalars[i]  # shape: [4]

            # Combine all scalars
            combined = torch.cat(
                [s.unsqueeze(0) for s in scalars_cp + scalars_seed] + [scalars_raw], dim=0)
            output_list.append(self.final(combined.unsqueeze(0)).squeeze())

        return torch.stack(output_list)

    
class TensorFlowDataset(Dataset):
    def __init__(self, all_data):
        self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Map string labels to integers
        label_map = {
            'local': 0,
            'not local': 1
        }
        label_str = item['label'].lower().strip()
        label_int = label_map[label_str]
    
        return (
            item['tensor'],  # [6, K, K]
            item['seed_ic'], item['seed_sir'],
            item['seed_ic_conv'], item['seed_sir_conv'],
            torch.tensor([item['contact_cost'], item['comm_cost'],
                          item['gamma'], item['behavioral_change_parameter']], dtype=torch.float32),
            torch.tensor(label_int, dtype=torch.float32)
        )

tl.set_backend('pytorch')

folder = '.'
instances = load_instances_from_folder(folder)

rank = 32 # Number of singular values to keep
cp_rank = 4
batch_size = 8
learning_rate = 0.001
num_epochs = 20

process_start_time = time.time()

all_tensor_inputs = []

for i, instance in enumerate(instances):

    print(f"Processing instance {i+1}/{len(instances)}: {instance}")

    tensor_input = build_tensor_input(instance, rank)
    all_tensor_inputs.append(tensor_input)
    
# save the tensor input to a .pkl file
with open(f'all_tensor_input_cluster.pkl', 'wb') as f:
    pickle.dump(all_tensor_inputs, f)
    
process_end_time = time.time()
print(f"Processing time: {process_end_time - process_start_time:.2f} seconds")
  
# load the tensor input from a .pkl file
with open('all_tensor_input_cluster.pkl', 'rb') as f:
    all_tensor_inputs = pickle.load(f)
  
imba_dataset = TensorFlowDataset(all_tensor_inputs)

minority_class = 0
majority_class = 1

minority_class_count = sum(1 for _, _, _, _, _, _, label in imba_dataset if label == minority_class)
majority_class_count = sum(1 for _, _, _, _, _, _, label in imba_dataset if label == majority_class)

print(f"Minority class count: {minority_class_count}")
print(f"Majority class count: {majority_class_count}")

repeat_factor = 5

minority_dataset = [item for item in imba_dataset if item[6] == minority_class]
majority_dataset = [item for item in imba_dataset if item[6] == majority_class]

minority_dataset_oversampled = minority_dataset * repeat_factor

data = minority_dataset_oversampled + majority_dataset

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_accuracies = []
k_fold_training_times = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    start_time = time.time()
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(data, batch_size=batch_size, sampler=test_subsampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = TensorCPClassifier(K=rank, cp_rank=cp_rank).to(device)
    model = InterpretableTensorClassifier(K=rank, cp_rank=cp_rank).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, label = batch
            tensor = tensor.to(device)
            seed_ic = seed_ic.to(device)
            seed_sir = seed_sir.to(device)
            seed_ic_conv = seed_ic_conv.to(device)
            seed_sir_conv = seed_sir_conv.to(device)
            scalars = scalars.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            logits = model(tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

    end_time = time.time()
    print(f'Training time for fold {fold+1}: {end_time - start_time:.2f} seconds')
    k_fold_training_times.append(end_time - start_time)

    # Validation
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, label = batch
            tensor = tensor.to(device)
            seed_ic = seed_ic.to(device)
            seed_sir = seed_sir.to(device)
            seed_ic_conv = seed_ic_conv.to(device)
            seed_sir_conv = seed_sir_conv.to(device)
            scalars = scalars.to(device)
            label = label.float().to(device)

            logits = model(tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars)
            preds = torch.sigmoid(logits) > 0.5

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    fold_acc = accuracy_score(val_labels, val_preds)
    fold_accuracies.append(fold_acc)
    print(f'fold {fold+1} accuracy: {fold_acc}')

print(f"Average accuracy across {k_folds} folds: {np.mean(fold_accuracies):.4f}") 
print(f"Standard deviation: {np.std(fold_accuracies):.4f}")
print(f"Average training time across {k_folds} folds: {np.mean(k_fold_training_times):.2f} seconds")
print(f"Standard deviation of training time: {np.std(k_fold_training_times):.2f} seconds")

    
    