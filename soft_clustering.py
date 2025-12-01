import os
import pickle
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.special import softmax
import markov_clustering as mc
from sklearn.mixture import GaussianMixture
# Constants
NUM_SUPERNODES = 32  # Set the fixed number of supernodes

# Helper functions
def load_instance(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def edge_list_to_adj_matrix(edge_list, starting, end):
    G = nx.Graph()
    G.add_nodes_from(range(starting, end))
    G.add_weighted_edges_from(edge_list)
    return nx.to_numpy_array(G)

def get_soft_cluster_assignments(adj_matrix, num_clusters):
    sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(adj_matrix)
    # Convert hard labels to soft assignments (one-hot for now, could be relaxed)
    soft_assignments = np.zeros((adj_matrix.shape[0], num_clusters))
    soft_assignments[np.arange(adj_matrix.shape[0]), labels] = 1
    return soft_assignments

def directed_svd_soft_clustering(adj_matrix, n_clusters=32):
    """
    Performs soft clustering on a directed graph using SVD and GMM.
    
    Parameters:
        adj_matrix (ndarray): The directed adjacency matrix (n x n)
        n_components (int): Number of singular values/vectors to keep
        n_clusters (int): Desired number of clusters
    
    Returns:
        soft_assignments (ndarray): Soft assignment matrix (n x n_clusters)
    """
    # Step 1: Compute truncated SVD: A ≈ U @ Σ @ V.T
    U, S, VT = np.linalg.svd(adj_matrix, full_matrices=False)
    
    # Sort singular values and vectors in descending order
    idx = np.argsort(-S)
    S = S[idx]
    U = U[:, idx]
    VT = VT[idx, :]

    # Step 2: Form node features by concatenating U and V
    Z = np.hstack((U, VT.T))  # Shape: (n x 2k)

    # Step 3: Fit a Gaussian Mixture Model for soft clustering
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
    gmm.fit(Z)
    
    # Step 4: Get soft assignment probabilities (responsibilities)
    soft_assignments = gmm.predict_proba(Z)  # Shape: (n x n_clusters)
    
    return soft_assignments

def pool_graph(adj_matrix, soft_assignments):
    return soft_assignments.T @ adj_matrix @ soft_assignments

def get_supernode_seed_vector(seed_vector, soft_assignments):
    return np.clip(soft_assignments.T @ seed_vector, 0, 1)

# Main processing function
def process_instance(file_path):
    data = load_instance(file_path)
    
    contact_edges = data['contact_network']  # list of (u, v, weight)
    comm_edges = data['communication_network']  # list of (u, v, weight)
    interlink = data['mapping']  # dict: comm_node -> contact_node

    label = data['label']
    contact_cost = data['contact_cost']
    comm_cost = data['communication_cost']
    gamma = data['gamma']
    behavioral_change_parameter = data['behavioral_change_parameter']

    # n_contact = max(u for u, _, _ in contact_edges) + 1
    # n_comm = max(v for _, v, _ in comm_edges) - n_contact + 1

    n_contact = max(max(u for u, _, _ in contact_edges), max(v for _, v, _ in contact_edges)) + 1
    n_comm = max(max(u for u, _, _ in comm_edges), max(v for _, v, _ in comm_edges)) - n_contact + 1

    A_contact = edge_list_to_adj_matrix(contact_edges, 0, n_contact)
    A_comm = edge_list_to_adj_matrix(comm_edges, n_contact, n_comm)

    # Interlink matrix
    A_interlink = np.zeros((n_contact, n_comm))
    for comm_node, contact_node in interlink.items():
        A_interlink[contact_node, comm_node-n_contact] = 1  # assuming binary connections

    # Clustering
    S_contact = directed_svd_soft_clustering(A_contact, NUM_SUPERNODES)
    S_comm = directed_svd_soft_clustering(A_comm, NUM_SUPERNODES)
    
    # S_contact = get_soft_cluster_assignments(A_contact, NUM_SUPERNODES)
    # S_comm = get_soft_cluster_assignments(A_comm, NUM_SUPERNODES)

    # for row in S_contact:
    #     print(row)
    #     print(sum(row))

    # Pooled (super) adjacency matrices
    super_contact = pool_graph(A_contact, S_contact)
    super_comm = pool_graph(A_comm, S_comm)
    super_interlink = S_contact.T @ A_interlink @ S_comm

    # Seed vector (binary): 1 if node is a source
    contact_seed_set = data['initial_infected']
    contact_seed = np.zeros(n_contact)
    for idx in contact_seed_set:
        contact_seed[idx] = 1
    comm_seed_set = data['initial_activated']
    comm_seed = np.zeros(n_comm)
    for idx in comm_seed_set:
        comm_seed[idx-n_contact] = 1

    super_contact_seed = get_supernode_seed_vector(contact_seed, S_contact)
    super_comm_seed = get_supernode_seed_vector(comm_seed, S_comm)

    return {
        "super_contact_adj": super_contact,
        "super_comm_adj": super_comm,
        "super_interlink_adj": super_interlink,
        "super_contact_seed": super_contact_seed,
        "super_comm_seed": super_comm_seed,
        "label": label,
        "contact_cost": contact_cost,
        "comm_cost": comm_cost,
        "gamma": gamma,
        "behavioral_change_parameter": behavioral_change_parameter
    }

# # # Sample test path
# test_instance_path = "results2_iter_0_beta_0.1_gamma_0.1_bp_3.pkl"
# processed = process_instance(test_instance_path)
# print(processed['super_contact_adj'].shape)
# print(processed['super_comm_adj'].shape)
# print(processed['super_interlink_adj'].shape)
# print(processed['super_contact_seed'].shape)
# print(processed['super_comm_seed'].shape)

