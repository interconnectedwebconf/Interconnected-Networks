import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import networkx as nx
import pickle
import numpy as np
import os
from util import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import time

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
    A_interlink = np.zeros((n_contact, n_comm))
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
    

def build_interlink_matrix(mapping, num_comm_nodes, num_contact_nodes):
    mat = np.zeros((num_comm_nodes, num_contact_nodes))
    for u_comm, u_contact in mapping.items():
        mat[u_comm][u_contact] = 1
    return mat

def build_supramatrix(A_contact, A_comm, A_interlink):
    
    top = np.hstack([A_contact, A_interlink])
    bottom = np.hstack([A_interlink.T, A_comm])
    S = np.vstack([top, bottom])
    return S

def process_instance2(instance):
    
    A_contact = instance['A_contact']
    A_comm = instance['A_comm']
    A_interlink = instance['A_interlink']
    contact_seed = instance['contact_seed']
    comm_seed = instance['comm_seed']
    both_seed = instance['both_seed']
    gamma = instance['gamma']
    behavioral_change_parameter = instance['behavioral_change_parameter']
    label = instance['label']

    S = build_supramatrix(A_contact, A_comm, A_interlink)

    # Flatten supramatrix and concatenate
    feature_vector = np.concatenate([
        S.flatten(),
        [gamma, behavioral_change_parameter]
    ])

    return feature_vector, label

# folder = '.'
# instances = load_instances_from_folder(folder)
# processed_instances = []

# for i, instance in enumerate(instances):
#     print(f"Processing instance {i+1}/{len(instances)}: {instance}")
#     processed_instance = process_instance(instance)
#     processed_instances.append(processed_instance)

# # save the processed instances to a .pkl file
# with open('all_processed_instances_supra.pkl', 'wb') as f:
#     pickle.dump(processed_instances, f)
    
# load the processed instances from a .pkl file
with open('all_processed_instances_supra.pkl', 'rb') as f:
    processed_instances = pickle.load(f)
    
# X, y = [], []
# for processed_instance in processed_instances:
#     feature, label = process_instance2(processed_instance)
#     X.append(feature)
#     y.append(label)
# X = np.array(X)
# y = np.array(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Classification using Random Forest
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print(classification_report(y_test, y_pred))

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)

class SupraGNNClassifier(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, graph_out_dim, scalar_dim):
        super().__init__()
        self.encoder = GraphEncoder(node_input_dim, hidden_dim, graph_out_dim)
        
        # MLP for final prediction
        combined_dim = graph_out_dim + scalar_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, supra_data, scalars):
        # Process the supra-adjacency graph
        graph_emb = self.encoder(supra_data.x, supra_data.edge_index, supra_data.batch)
        
        # Combine with scalar features
        combined = torch.cat([graph_emb, scalars], dim=1)
        # Ensure output has shape (batch_size, 1)
        return self.mlp(combined).view(-1, 1)

def convert_to_pyg_data(adj_matrix, node_features=None):
    # Convert adjacency matrix to edge_index format
    edge_index = []
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    if node_features is None:
        # Use degree as node feature if none provided
        node_features = np.sum(adj_matrix, axis=1)
    
    x = torch.tensor(node_features, dtype=torch.float).reshape(-1, 1)
    
    return Data(x=x, edge_index=edge_index)

class SupraDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        # Build supra-adjacency matrix
        S = build_supramatrix(
            instance['A_contact'],
            instance['A_comm'],
            instance['A_interlink']
        )
        
        # Combine seed vectors for node features
        node_features = instance['both_seed']
        
        # Convert to PyG Data object
        supra_data = convert_to_pyg_data(S, node_features)
        
        # Prepare scalar features and label
        scalars = torch.tensor([
            instance['gamma'],
            instance['behavioral_change_parameter']
        ], dtype=torch.float32)
        
        label_map = {'local': 0, 'not local': 1}
        label = label_map[instance['label'].lower().strip()]
        
        return supra_data, scalars, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    supra_graphs = [item[0] for item in batch]
    scalars = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    
    supra_batch = Batch.from_data_list(supra_graphs)
    
    return supra_batch, scalars, labels

# Model parameters
hidden_dim = 64
graph_out_dim = 32
scalar_dim = 2
num_epochs = 20
batch_size = 8
lr = 0.001

# Create dataset
imba_dataset = SupraDataset(processed_instances)

minority_class = 0
majority_class = 1

minority_class_count = sum(1 for _, _, label in imba_dataset if label == minority_class)
majority_class_count = sum(1 for _, _, label in imba_dataset if label == majority_class)

print(f"Minority class count: {minority_class_count}")
print(f"Majority class count: {majority_class_count}")

repeat_factor = 5

minority_dataset = [item for item in imba_dataset if item[2] == minority_class]
majority_dataset = [item for item in imba_dataset if item[2] == majority_class]

minority_dataset_oversampled = minority_dataset * repeat_factor

dataset = minority_dataset_oversampled + majority_dataset
# dataset = dataset.shuffle(buffer_size=len(dataset))

print(f"New dataset size: {len(dataset)}")


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# K-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_accuracies = []
k_fold_training_times = []

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    start_time = time.time()
    print(f"\nFold {fold+1}/{k_folds}")
    
    # Create data loaders
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = SupraGNNClassifier(
        node_input_dim=1,
        hidden_dim=hidden_dim,
        graph_out_dim=graph_out_dim,
        scalar_dim=scalar_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for supra_batch, scalars, labels in train_loader:
            # Move to device
            supra_batch = supra_batch.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device).view(-1, 1)  # Reshape labels to (batch_size, 1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(supra_batch, scalars)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            avg_loss = np.mean(train_losses)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    print(f'Training time for fold {fold+1}: {end_time - start_time:.2f} seconds')
    k_fold_training_times.append(end_time - start_time)

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for supra_batch, scalars, labels in test_loader:
            supra_batch = supra_batch.to(device)
            scalars = scalars.to(device)
            labels = labels.to(device).view(-1, 1)  # Reshape labels consistently
            
            outputs = model(supra_batch, scalars)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().detach().numpy().flatten())
            all_labels.extend(labels.cpu().detach().numpy().flatten())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    fold_accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

print("\nOverall Results:")
print(f"Average accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies) * 2:.4f})")
print(f"Average training time across {k_folds} folds: {np.mean(k_fold_training_times):.2f} seconds")
print(f"Standard deviation of training time: {np.std(k_fold_training_times):.2f} seconds")
