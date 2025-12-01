import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GINConv, GATConv
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import networkx as nx
import pickle
import numpy as np
import os
from util import *

def process_instance(file_path):
    data = load_instance(file_path)
    
    contact_edges = data['contact_network']
    comm_edges = data['communication_network']
    interlink = data['mapping']
    
    label = data['label']
    gamma = data['gamma']
    behavioral_change_parameter = data['behavioral_change_parameter']
    
    n_contact = max(max(u for u, _, _ in contact_edges), max(v for _, v, _ in contact_edges)) + 1
    n_comm = max(max(u for u, _, _ in comm_edges), max(v for _, v, _ in comm_edges)) - n_contact + 1

    A_contact = edge_list_to_adj_matrix(contact_edges, 0, n_contact)
    A_comm = edge_list_to_adj_matrix(comm_edges, n_contact, n_comm)

    # Interlink matrix
    A_interlink = np.zeros((n_contact+n_comm, n_contact+n_comm))
    for comm_node, contact_node in interlink.items():
        A_interlink[contact_node, comm_node-n_contact] = 1  # assuming binary connections
    
    contact_seed_set = data['initial_infected']
    comm_seed_set = data['initial_activated']
    contact_seed = np.zeros(n_contact)
    comm_seed = np.zeros(n_comm)
    both_seed = np.zeros(n_contact+n_comm)
    for idx in contact_seed_set:
        contact_seed[idx] = 1
    for idx in comm_seed_set:
        comm_seed[idx-n_contact] = 1
    for idx in contact_seed_set:
        both_seed[idx] = 1
    for idx in comm_seed_set:
        both_seed[n_contact] = 1
    
    return {
        'A_contact': A_contact,
        'A_comm': A_comm,
        'A_interlink': A_interlink,
        'contact_seed': contact_seed,
        'comm_seed': comm_seed,
        'both_seed': both_seed,
        'label': label,
        'gamma': gamma,
        'behavioral_change_parameter': behavioral_change_parameter
    }

class GCN_Encoder(nn.Module):
    # GCN
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # graph-level embedding

class GIN_Encoder(nn.Module):
    # GIN
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN_Encoder, self).__init__()
        # For GINConv, we need to define the nn module first
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Then create GINConv layers with the nn modules
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # graph-level embedding

class GAT_Encoder(nn.Module):
    # GAT
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # graph-level embedding
        
        
class GNNClassifier(nn.Module):
    # GCN_Encoder, GIN_Encoder, GAT_Encoder
    def __init__(self, node_input_dim, hidden_dim, graph_out_dim, scalar_dim):
        super(GNNClassifier, self).__init__()
        self.gnn_contact = GAT_Encoder(node_input_dim, hidden_dim, graph_out_dim)
        self.gnn_comm = GAT_Encoder(node_input_dim, hidden_dim, graph_out_dim)
        self.gnn_interlink = GAT_Encoder(node_input_dim, hidden_dim, graph_out_dim)

        # MLP for final prediction
        combined_dim = 3 * graph_out_dim + scalar_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, contact_batch, comm_batch, interlink_batch, scalars):
        # Process each graph type
        contact_emb = self.gnn_contact(contact_batch.x, contact_batch.edge_index, contact_batch.batch)
        comm_emb = self.gnn_comm(comm_batch.x, comm_batch.edge_index, comm_batch.batch)
        interlink_emb = self.gnn_interlink(interlink_batch.x, interlink_batch.edge_index, interlink_batch.batch)
        
        # Combine embeddings with scalars
        combined = torch.cat([contact_emb, comm_emb, interlink_emb, scalars], dim=1)
        return self.mlp(combined).squeeze()

def convert_to_pyg_data(adj_matrix, node_features):
    # Convert adjacency matrix to edge_index format
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    x = torch.tensor(node_features, dtype=torch.float).reshape(-1, 1)
    
    return Data(x=x, edge_index=edge_index)

class TensorFlowDataset(Dataset):
    def __init__(self, all_data):
        self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert adjacency matrices to PyG Data objects
        contact_data = convert_to_pyg_data(
            item['A_contact'], 
            item['contact_seed']
        )
        comm_data = convert_to_pyg_data(
            item['A_comm'], 
            item['comm_seed']
        )
        interlink_data = convert_to_pyg_data(
            item['A_interlink'], 
            item['both_seed']
        )
        
        # Map string labels to integers
        label_map = {
            'local': 0,
            'not local': 1
        }
        label_str = item['label'].lower().strip()
        label_int = label_map[label_str]
    
        return (
            contact_data,
            comm_data,
            interlink_data,
            torch.tensor([item['gamma'], item['behavioral_change_parameter']], dtype=torch.float32),
            torch.tensor(label_int, dtype=torch.float32)
        )

def collate_fn(batch):
    # Separate the different components
    contact_graphs = [item[0] for item in batch]
    comm_graphs = [item[1] for item in batch]
    interlink_graphs = [item[2] for item in batch]
    scalars = torch.stack([item[3] for item in batch])
    labels = torch.stack([item[4] for item in batch])
    
    # Batch the graphs
    contact_batch = Batch.from_data_list(contact_graphs)
    comm_batch = Batch.from_data_list(comm_graphs)
    interlink_batch = Batch.from_data_list(interlink_graphs)
    
    return contact_batch, comm_batch, interlink_batch, scalars, labels

# Model parameters
hidden_dim = 64
graph_out_dim = 128
scalar_dim = 2
num_epochs = 20
batch_size = 8
lr = 0.001

# process_start_time = time.time()
# folder = '.'
# instances = load_instances_from_folder(folder)

# all_processed_instances = []

# for i, instance in enumerate(instances):
    
#     print(f"Processing instance {i+1}/{len(instances)}: {instance}")
#     processed_instance = process_instance(instance)
#     all_processed_instances.append(processed_instance)
    
# # save the processed instances to a .pkl file
# with open('all_processed_instances_gnn.pkl', 'wb') as f:
#     pickle.dump(all_processed_instances, f)

# process_end_time = time.time()
# print(f"Time taken to process all instances: {process_end_time - process_start_time:.2f} seconds")

# Load processed instances
with open('all_processed_instances_gnn.pkl', 'rb') as f:
    all_processed_instances = pickle.load(f)

# Create dataset
imba_dataset = TensorFlowDataset(all_processed_instances)

minority_class = 0
majority_class = 1

minority_class_count = sum(1 for _, _, _, _, label in imba_dataset if label == minority_class)
majority_class_count = sum(1 for _, _, _, _, label in imba_dataset if label == majority_class)

print(f"Minority class count: {minority_class_count}")
print(f"Majority class count: {majority_class_count}")

repeat_factor = 5

minority_dataset = [item for item in imba_dataset if item[4] == minority_class]
majority_dataset = [item for item in imba_dataset if item[4] == majority_class]

minority_dataset_oversampled = minority_dataset * repeat_factor

data = minority_dataset_oversampled + majority_dataset

print(len(data))

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_accuracies = []
k_fold_training_times = []

for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
    start_time = time.time()
    
    train_subsampler = SubsetRandomSampler(train_idx)
    test_subsampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(
        data, 
        batch_size=batch_size, 
        sampler=train_subsampler,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        data, 
        batch_size=batch_size, 
        sampler=test_subsampler,
        collate_fn=collate_fn
    )

    # Initialize model
    model = GNNClassifier(
        node_input_dim=1,
        hidden_dim=hidden_dim,
        graph_out_dim=graph_out_dim,
        scalar_dim=scalar_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for contact_batch, comm_batch, interlink_batch, scalars, labels in train_loader:
            # Move to device
            contact_batch = contact_batch.to(device)
            comm_batch = comm_batch.to(device)
            interlink_batch = interlink_batch.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device).reshape(-1, 1)
            
            optimizer.zero_grad()
            pred = model(contact_batch, comm_batch, interlink_batch, scalars)
            pred = pred.view(-1, 1)
            loss = F.binary_cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
        if (epoch + 1) % 10 == 0:
            print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(train_losses):.4f}")

    end_time = time.time()
    print(f'Training time for fold {fold+1}: {end_time - start_time:.2f} seconds')
    k_fold_training_times.append(end_time - start_time)

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for contact_batch, comm_batch, interlink_batch, scalars, labels in test_loader:
            # Move to device
            contact_batch = contact_batch.to(device)
            comm_batch = comm_batch.to(device)
            interlink_batch = interlink_batch.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device)
            
            pred = model(contact_batch, comm_batch, interlink_batch, scalars)
            pred_labels = (pred > 0.5).float()
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    fold_accuracies.append(accuracy)
    print(f"Fold {fold+1}/{k_folds} - Accuracy: {accuracy:.4f}")

print(f"Average accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies) * 2:.4f})")
print(f"Average training time across {k_folds} folds: {np.mean(k_fold_training_times):.2f} seconds")
print(f"Standard deviation of training time: {np.std(k_fold_training_times):.2f} seconds")





