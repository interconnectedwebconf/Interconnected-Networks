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
import pickle
from scipy.sparse.linalg import eigsh
import os
import csv

def spectral_kernel_loss(value, matrix):
    # Calculate the spectral properties (eigenvalues) of both matrices
    eigenvalues = torch.linalg.eigvalsh(matrix)
    
    # Sort eigenvalues in ascending order
    eigenvalues, _ = torch.sort(eigenvalues)
    
    # Compare multiple eigenvalues
    if value and eigenvalues.size(0) > 1:
        # Compare first few eigenvalues
        k = min(5, eigenvalues.size(0))
        eig_loss = torch.mean(torch.abs(value - eigenvalues[1:k]))
        
        # Add regularization to encourage non-identity structure
        identity = torch.eye(matrix.size(0), device=matrix.device)
        reg_loss = torch.mean(torch.abs(matrix - identity))
        
        # Add a term to encourage the matrix to be different from identity
        # This should help gradients flow to the S matrices
        diff_from_identity = torch.mean(torch.abs(matrix - identity))
        
        # Add a term to encourage the matrix to have non-zero off-diagonal elements
        off_diag = matrix.clone()
        off_diag.diagonal().zero_()
        off_diag_loss = -torch.mean(torch.abs(off_diag))  # Negative because we want to maximize
        
        return eig_loss + 0.1 * reg_loss + 0.1 * diff_from_identity + 0.1 * off_diag_loss
    else:
        return torch.tensor(0.0, device=matrix.device)


def reconstruct_matrix_from_svd(A, k):
    """Reconstructs a matrix from its top-k singular components."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Sort singular values and vectors in descending order
    idx = np.argsort(-S)
    S = S[idx]
    U = U[:, idx]
    Vt = Vt[idx, :]
    
    U = U[:, :k]            # [n, 32]
    S = S[:k]               # [32]
    Vt = Vt[:k, :]          # [32, n]
    
    S = np.diag(S)
    S_sqrt = np.sqrt(S)

    Z1 = U @ S_sqrt
    Z2 = S_sqrt @ Vt
    
    return U, Vt, Z1, Z2

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
    instance = load_instance(file_path)
    
    contact_edges = instance['contact_network']  # list of (u, v, weight)
    comm_edges = instance['communication_network']  # list of (u, v, weight)
    interlink = instance['mapping']  # dict: comm_node -> contact_node
    
    n_contact = max(max(u for u, _, _ in contact_edges), max(v for _, v, _ in contact_edges)) + 1
    n_comm = max(max(u for u, _, _ in comm_edges), max(v for _, v, _ in comm_edges)) - n_contact + 1
    
    A_contact = edge_list_to_adj_matrix(contact_edges, 0, n_contact)
    A_comm = edge_list_to_adj_matrix(comm_edges, n_contact, n_comm)

    # Interlink matrix
    A_interlink = np.zeros((n_comm+n_contact, n_comm+n_contact))
    for comm_node, contact_node in interlink.items():
        A_interlink[comm_node, contact_node] = 1  
         
    label = instance['label']
    contact_cost = instance['contact_cost']
    comm_cost = instance['communication_cost']
    gamma = instance['gamma']
    behavioral_change_parameter = instance['behavioral_change_parameter']
    
    U_ic, V_ic, Z1_ic, Z2_ic = reconstruct_matrix_from_svd(A_comm, rank)
    U_sir, V_sir, Z1_sir, Z2_sir = reconstruct_matrix_from_svd(A_contact, rank)
    
    appro_ic_1 = Z1_ic.T @ A_comm @ Z1_ic
    appro_ic_2 = Z2_ic @ A_comm.T @ Z2_ic.T
    appro_sir_1 = Z1_sir.T @ A_contact @ Z1_sir
    appro_sir_2 = Z2_sir @ A_contact.T @ Z2_sir.T
    
    # Consider three ways to construct the interlink matrix
    # 1. Use the interlink matrix to do svd
    # 2. Use the sum of the product of the node weight in each component as the weight of the interlink
    # 3. Use U and V from the svd of the contact and communication matrices to do reconstructio
    
    # 1. Use the interlink matrix to do svd
    _, _, Z1_inter_svd, Z2_inter_svd = reconstruct_matrix_from_svd(A_interlink, rank)
    Z1_inter_reconstruct = Z1_inter_svd.T @ (A_interlink) @ Z1_inter_svd
    Z2_inter_reconstruct = Z2_inter_svd @ (A_interlink) @ Z2_inter_svd.T
    
    # second eigenvalue of the interlink matrix
    second_eig_interlink_out = np.linalg.eigvalsh(Z1_inter_reconstruct)[1]
    second_eig_interlink_in = np.linalg.eigvalsh(Z2_inter_reconstruct)[1]
    # # 2. Use the sum of the product of the node weight in each component as the weight of the interlink
    # Z1_inter_sum = np.zeros((rank, rank))
    # Z2_inter_sum = np.zeros((rank, rank))
    # for i in range(rank):
    #     for j in range(rank):
    #         Z1_inter_sum[i, j] = 0
    #         for comm_node, contact_node in interlink.items():
    #             Z1_inter_sum[i, j] += [comm_node, i] * Z1_sir[contact_node, j]
    
    
    # # 3. Use U and V from the svd of the contact and communication matrices to do reconstruction
    # Z1_inter_reconstruct = U_sir.T @ (A_interlink.T) @ V_ic.T
    # Z2_inter_reconstruct = U_ic.T @ A_interlink @ V_sir.T
    

    # Stack into a tensor: shape [6, K, K]
    tensor_input1 = np.stack([appro_ic_1, appro_ic_2, appro_sir_1, appro_sir_2, Z1_inter_reconstruct, Z2_inter_reconstruct])
    # tensor_input2 = np.stack([appro_ic_1, appro_ic_2, appro_sir_1, appro_sir_2, appro_inter_1, appro_inter_2])
    
    # Include seed vectors as separate features (optional)
    seed_ic_set = instance['initial_activated']  # shape [n_comm]
    seed_ic = np.zeros(n_comm)
    for idx in seed_ic_set:
        seed_ic[idx-n_contact] = 1
    seed_sir_set = instance['initial_infected']  # shape [n_contact]
    seed_sir = np.zeros(n_contact)
    for idx in seed_sir_set:
        seed_sir[idx] = 1
        
    component_seed_ic = Z1_ic.T @ seed_ic
    component_seed_sir = Z1_sir.T @ seed_sir  
    
    seed_ic_conv = appro_ic_1 @ component_seed_ic
    seed_sir_conv = appro_sir_1 @ component_seed_sir

    return {
        'tensor1': tensor_input1,        # shape [6, K, K]
        # 'tensor_baseline': tensor_input2,        # shape [6, K, K]
        'seed_ic': component_seed_ic,            # shape [K]
        'seed_sir': component_seed_sir,           # shape [K]
        'seed_ic_conv': seed_ic_conv,  # shape [K]
        'seed_sir_conv': seed_sir_conv,  # shape [K]
        'label': label,                # label
        # 'contact_cost': contact_cost,  # scaler 
        # 'comm_cost': comm_cost,        # scaler 
        'gamma': gamma,                # scaler 
        'behavioral_change_parameter': behavioral_change_parameter,  # scaler
        'second_eig_interlink_out': second_eig_interlink_out,  # scaler
        'second_eig_interlink_in': second_eig_interlink_in  # scaler
    }, n_contact, n_comm


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

class CPDecomposition(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        
    def forward(self, tensor):
        # Get the device and shape
        device = tensor.device
        K = tensor.shape[1]
        
        # Initialize factors as learnable parameters
        A = nn.Parameter(torch.randn(6, self.rank, device=device))
        B = nn.Parameter(torch.randn(K, self.rank, device=device))
        C = nn.Parameter(torch.randn(K, self.rank, device=device))
        
        # Normalize factors
        A.data = F.normalize(A.data, dim=0)
        B.data = F.normalize(B.data, dim=0)
        C.data = F.normalize(C.data, dim=0)
        
        # Reconstruct tensor
        reconstructed = torch.zeros_like(tensor)
        for r in range(self.rank):
            # Compute outer product for each component
            outer = torch.einsum('i,j,k->ijk', A[:, r], B[:, r], C[:, r])
            reconstructed += outer
        
        # Compute reconstruction loss
        loss = F.mse_loss(reconstructed, tensor)
        
        return [A, B, C], loss

class MatrixTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.save_for_backward(x, s)
        return x @ s

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        grad_x = grad_output @ s.T
        grad_s = x.T @ grad_output
        return grad_x, grad_s

class InterpretableTensorClassifier(nn.Module):
    def __init__(self, K, cp_rank, hidden=32):
        super().__init__()
        self.cp_rank = cp_rank
        self.K = K

        # Learnable matrices for interlink layer - initialize with small random values
        self.S1 = nn.Parameter(torch.eye(K, K) + 0.1 * torch.randn(K, K))
        self.S2 = nn.Parameter(torch.eye(K, K) + 0.1 * torch.randn(K, K))
        self.S3 = nn.Parameter(torch.eye(K, K) + 0.1 * torch.randn(K, K))
        self.S4 = nn.Parameter(torch.eye(K, K) + 0.1 * torch.randn(K, K))

        # CP decomposition module
        self.cp_decomposition = CPDecomposition(cp_rank)

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
            nn.Linear(cp_rank + 6, hidden),  # 6 = 4 seed scalars + 2 raw scalars
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, tensor_input, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, second_eig_interlink_out, second_eig_interlink_in):
        # Ensure all tensors are float32 and require gradients
        tensor_input = tensor_input.float().requires_grad_(True)
        seed_ic = seed_ic.float().requires_grad_(True)
        seed_sir = seed_sir.float().requires_grad_(True)
        seed_ic_conv = seed_ic_conv.float().requires_grad_(True)
        seed_sir_conv = seed_sir_conv.float().requires_grad_(True)
        scalars = scalars.float().requires_grad_(True)
        second_eig_interlink_out = second_eig_interlink_out.float().requires_grad_(True)
        second_eig_interlink_in = second_eig_interlink_in.float().requires_grad_(True)
        
        B = tensor_input.size(0)
        output_list = []
        spectral_loss_out_list = []
        spectral_loss_in_list = []
        cp_loss_list = []
        s_matrix_loss_list = []

        for i in range(B):
            appro_ic_1 = tensor_input[i, 0]
            appro_ic_2 = tensor_input[i, 1]
            appro_sir_1 = tensor_input[i, 2]
            appro_sir_2 = tensor_input[i, 3]
            Z1_inter_reconstruct = tensor_input[i, 4]
            Z2_inter_reconstruct = tensor_input[i, 5]

            # Ensure input matrices require gradients
            appro_ic_1.requires_grad_(True)
            appro_ic_2.requires_grad_(True)
            appro_sir_1.requires_grad_(True)
            appro_sir_2.requires_grad_(True)
            Z1_inter_reconstruct.requires_grad_(True)
            Z2_inter_reconstruct.requires_grad_(True)

            # Use custom matrix multiplication with explicit gradient computation
            # For Z1_inter
            temp1 = MatrixTransform.apply(appro_ic_1, self.S1)
            temp2 = MatrixTransform.apply(temp1, Z1_inter_reconstruct)
            temp3 = MatrixTransform.apply(temp2, self.S2)
            Z1_inter = MatrixTransform.apply(temp3, appro_sir_2.T)

            # For Z2_inter
            temp4 = MatrixTransform.apply(appro_sir_1, self.S3)
            temp5 = MatrixTransform.apply(temp4, Z2_inter_reconstruct)
            temp6 = MatrixTransform.apply(temp5, self.S4)
            Z2_inter = MatrixTransform.apply(temp6, appro_ic_2.T)

            # Add direct loss terms for S matrices that encourage meaningful transformations
            identity = torch.eye(self.K, device=self.S1.device)
            
            # Encourage S matrices to be different from identity
            s1_diff_loss = -torch.mean(torch.abs(self.S1 - identity))  # Negative because we want to maximize difference
            s2_diff_loss = -torch.mean(torch.abs(self.S2 - identity))
            s3_diff_loss = -torch.mean(torch.abs(self.S3 - identity))
            s4_diff_loss = -torch.mean(torch.abs(self.S4 - identity))
            
            # Encourage non-zero off-diagonal elements
            s1_off_diag = self.S1.clone()
            s1_off_diag.diagonal().zero_()
            s1_off_diag_loss = -torch.mean(torch.abs(s1_off_diag))
            
            s2_off_diag = self.S2.clone()
            s2_off_diag.diagonal().zero_()
            s2_off_diag_loss = -torch.mean(torch.abs(s2_off_diag))
            
            s3_off_diag = self.S3.clone()
            s3_off_diag.diagonal().zero_()
            s3_off_diag_loss = -torch.mean(torch.abs(s3_off_diag))
            
            s4_off_diag = self.S4.clone()
            s4_off_diag.diagonal().zero_()
            s4_off_diag_loss = -torch.mean(torch.abs(s4_off_diag))
            
            # Combine S matrix losses
            s_matrix_loss = (s1_diff_loss + s2_diff_loss + s3_diff_loss + s4_diff_loss + 
                           s1_off_diag_loss + s2_off_diag_loss + s3_off_diag_loss + s4_off_diag_loss)
            s_matrix_loss_list.append(s_matrix_loss)

            # Add a direct loss term that depends on the transformed matrices
            transform_loss = torch.mean(torch.abs(Z1_inter - Z1_inter_reconstruct)) + torch.mean(torch.abs(Z2_inter - Z2_inter_reconstruct))
            s_matrix_loss_list.append(transform_loss)

            spectral_loss_out = spectral_kernel_loss(second_eig_interlink_out[i], Z1_inter)
            spectral_loss_in = spectral_kernel_loss(second_eig_interlink_in[i], Z2_inter)
            spectral_loss_out_list.append(spectral_loss_out)
            spectral_loss_in_list.append(spectral_loss_in)

            # Create a new tensor for the updated tensor_input
            updated_tensor_input = tensor_input.clone()
            updated_tensor_input[i, 4] = Z1_inter
            updated_tensor_input[i, 5] = Z2_inter

            # Use CP decomposition
            factors, cp_loss = self.cp_decomposition(updated_tensor_input[i])
            cp_loss_list.append(cp_loss)

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
            scalars_raw = scalars[i]  # shape: [2]

            # Combine all scalars
            combined = torch.cat(
                [s.unsqueeze(0) for s in scalars_cp + scalars_seed] + [scalars_raw], dim=0)
            output_list.append(self.final(combined.unsqueeze(0)).squeeze())

        return torch.stack(output_list), torch.stack(spectral_loss_out_list).mean(), torch.stack(spectral_loss_in_list).mean(), torch.stack(cp_loss_list).mean(), torch.stack(s_matrix_loss_list).mean()

    
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
            item['tensor1'],  # [6, K, K]
            item['seed_ic'], item['seed_sir'],
            item['seed_ic_conv'], item['seed_sir_conv'],
            torch.tensor([item['gamma'], item['behavioral_change_parameter']], dtype=torch.float32),
            torch.tensor(label_int, dtype=torch.float32),
            item['second_eig_interlink_out'], item['second_eig_interlink_in']
        )

tl.set_backend('pytorch')

folder = '.'
instances = load_instances_from_folder(folder)

rank = 32 # Number of singular values to keep
cp_rank = 4
batch_size = 8
learning_rate = 0.01  # Increased learning rate
num_epochs = 20  # Increased number of epochs

# process_start_time = time.time()
# all_tensor_inputs = []
# instance_stats = []

# for i, instance in enumerate(instances):

#     print(f"Processing instance {i+1}/{len(instances)}: {instance}")

#     t0 = time.time()

#     tensor_input, n_contact, n_comm = build_tensor_input(instance, rank)
#     all_tensor_inputs.append(tensor_input)
    
#     t1 = time.time()
#     elapsed_time = t1 - t0
    
#     instance_stats.append({
#         'n_contact': n_contact,
#         'n_comm': n_comm,
#         'elapsed_time': elapsed_time
#     })
    
# # save the tensor input to a .pkl file
# with open(f'all_tensor_input_recons_loss.pkl', 'wb') as f:
#     pickle.dump(all_tensor_inputs, f)

# process_end_time = time.time()
# print(f"Processing time: {process_end_time - process_start_time:.2f} seconds")

# # save the instance stats to a .csv file
# with open(f'instance_stats_recons_loss.csv', 'w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=['n_contact', 'n_comm', 'elapsed_time'])
#     writer.writeheader()
#     writer.writerows(instance_stats)

# # load the tensor input from a .pkl file
# with open('all_tensor_input_recons_loss.pkl', 'rb') as f:
#     all_tensor_inputs = pickle.load(f)
  
# imba_dataset = TensorFlowDataset(all_tensor_inputs)

# minority_class = 0
# majority_class = 1

# minority_class_count = sum(1 for _, _, _, _, _, _, label, _, _ in imba_dataset if label == minority_class)
# majority_class_count = sum(1 for _, _, _, _, _, _, label, _, _ in imba_dataset if label == majority_class)

# print(f"Minority class count: {minority_class_count}")
# print(f"Majority class count: {majority_class_count}")

# repeat_factor = 5

# minority_dataset = [item for item in imba_dataset if item[6] == minority_class]
# majority_dataset = [item for item in imba_dataset if item[6] == majority_class]

# minority_dataset_oversampled = minority_dataset * repeat_factor

# data = minority_dataset_oversampled + majority_dataset

# print(len(data))

# # # explore the distribution of the data label, the percentage of local and not local
# # label_distribution = [item['label'] for item in all_tensor_inputs]
# # local_percentage = label_distribution.count('Local') / len(label_distribution)
# # not_local_percentage = label_distribution.count('Not Local') / len(label_distribution)
# # print(f"Local percentage: {local_percentage:.2%}")
# # print(f"Not local percentage: {not_local_percentage:.2%}")

# k_folds = 5
# kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# k_fold_accuracies = []
# k_fold_training_times = []
# for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
#     start_time = time.time()
    
#     train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
#     test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

#     train_loader = DataLoader(data, batch_size=batch_size, sampler=train_subsampler)
#     test_loader = DataLoader(data, batch_size=batch_size, sampler=test_subsampler)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = InterpretableTensorClassifier(K=rank, cp_rank=cp_rank).to(device)

#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         model.train()
#         train_losses = []

#         for batch in train_loader:
#             tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, label, second_eig_interlink_out, second_eig_interlink_in = batch
#             tensor = tensor.to(device)
#             seed_ic = seed_ic.to(device)
#             seed_sir = seed_sir.to(device)
#             seed_ic_conv = seed_ic_conv.to(device)
#             seed_sir_conv = seed_sir_conv.to(device)
#             scalars = scalars.to(device)
#             label = label.float().to(device)
#             second_eig_interlink_out = second_eig_interlink_out.to(device)
#             second_eig_interlink_in = second_eig_interlink_in.to(device)

#             optimizer.zero_grad()
#             logits, spectral_loss_in, spectral_loss_out, cp_loss, s_matrix_loss = model(tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, second_eig_interlink_out, second_eig_interlink_in)
#             classification_loss = criterion(logits, label)
            
#             # Add direct losses for S matrices
#             s1_loss = torch.mean(torch.abs(model.S1 - torch.eye(model.K, device=device)))
#             s2_loss = torch.mean(torch.abs(model.S2 - torch.eye(model.K, device=device)))
#             s3_loss = torch.mean(torch.abs(model.S3 - torch.eye(model.K, device=device)))
#             s4_loss = torch.mean(torch.abs(model.S4 - torch.eye(model.K, device=device)))
            
#             loss = classification_loss + 0.1 * spectral_loss_in + 0.1 * spectral_loss_out + 0.1 * cp_loss + 0.1 * (s1_loss + s2_loss + s3_loss + s4_loss + s_matrix_loss)
#             loss.backward()

#             # Print gradients after backward pass
#             print(f"After backward:")
#             print(f"S1 grad norm: {model.S1.grad.norm().item() if model.S1.grad is not None else 'None'}")
#             print(f"S2 grad norm: {model.S2.grad.norm().item() if model.S2.grad is not None else 'None'}")
#             print(f"S3 grad norm: {model.S3.grad.norm().item() if model.S3.grad is not None else 'None'}")
#             print(f"S4 grad norm: {model.S4.grad.norm().item() if model.S4.grad is not None else 'None'}")

#             optimizer.step()

#             train_losses.append(loss.item())

#     end_time = time.time()
#     print(f'Training time for fold {fold+1}: {end_time - start_time:.2f} seconds')
#     k_fold_training_times.append(end_time - start_time)

#     # Validation
#     model.eval()
#     val_preds, val_labels = [], []

#     with torch.no_grad():
#         for batch in test_loader:
#             tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, label, second_eig_interlink_out, second_eig_interlink_in = batch
#             tensor = tensor.to(device)
#             seed_ic = seed_ic.to(device)
#             seed_sir = seed_sir.to(device)
#             seed_ic_conv = seed_ic_conv.to(device)
#             seed_sir_conv = seed_sir_conv.to(device)
#             scalars = scalars.to(device)
#             label = label.float().to(device)

#             logits,_,_,_,_ = model(tensor, seed_ic, seed_sir, seed_ic_conv, seed_sir_conv, scalars, second_eig_interlink_out, second_eig_interlink_in)
#             preds = torch.sigmoid(logits) > 0.5

#             val_preds.extend(preds.cpu().numpy())
#             val_labels.extend(label.cpu().numpy())
    
#     accuracy = accuracy_score(val_labels, val_preds)
#     k_fold_accuracies.append(accuracy)
#     print(f'fold {fold+1} accuracy: {accuracy}')

# print(f"Average accuracy across {k_folds} folds: {np.mean(k_fold_accuracies):.4f}")
# print(f"Standard deviation: {np.std(k_fold_accuracies):.4f}")
# print(f"Average training time across {k_folds} folds: {np.mean(k_fold_training_times):.2f} seconds")
# print(f"Standard deviation of training time: {np.std(k_fold_training_times):.2f} seconds")

# # Save the last trained model
# torch.save(model.state_dict(), 'interpretable_tensor_model.pth')
# print("Model saved as 'interpretable_tensor_model.pth'")


