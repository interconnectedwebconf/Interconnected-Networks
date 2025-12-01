import networkx as nx
import random
from copy import deepcopy
import numpy as np
import pickle
from sklearn.model_selection import KFold
import os
import time

def load_instances_from_folder(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')]
    instances = []
    for file_path in file_paths:
        if file_path.startswith('./results_iter') or file_path.startswith('./all_'):
            continue
        instances.append(file_path)
    return instances

def sir_model_step(graph, infected, beta_values, recovered, ic_activated_nodes, beta, neighbors, mapping, gamma=0.1, behavioral_change_parameter = 3):
    """
    Simulates one step of the SIR process with the given communication network influence (IC) and infection spread (SIR).

    Parameters:
        graph (nx.Graph): The graph representing the epidemic network.
        infected (set): Set of infected nodes.
        beta_values (dict): Dictionary of infection rates (beta values) for each node.
        recovered (set): Set of recovered nodes.
        ic_activated_nodes (set): Set of activated nodes in the communication network (IC model).
        communication_graph (nx.Graph): The communication network (IC model).
        neighbors (dict): Dictionary of neighbors for each node in the epidemic graph.
        gamma (float): Recovery rate.

    Returns:
        set: Set of newly infected nodes after this step.
        set: Set of newly recovered nodes after this step.
    """
    new_infected = set()
    new_recovered = set()

    # Update beta values based on communication network activations
    for activated_node in ic_activated_nodes:
        
        human = mapping[activated_node]
        if human in graph.nodes():
            # Increase the beta value of the corresponding node in the epidemic network
            beta_values[human] = min(behavioral_change_parameter * beta, 1)  # As per the condition of IC affecting SIR

    # Spread the infection: check each infected node's neighbors
    for node in infected:
        for neighbor in neighbors[node]:
            if neighbor not in infected and neighbor not in recovered:
                # Infection occurs if the random threshold is less than the beta value
                if random.random() < beta_values[node]:
                    new_infected.add(neighbor)

        # Recover infected nodes probabilistically based on recovery rate (gamma)
    for node in infected:
        if node not in recovered:
            if random.random() < gamma:
                new_recovered.add(node)

    # Return the sets of newly infected and newly recovered nodes
    return new_infected, new_recovered

def run_joint_diffusion(graph, communication_graph, initial_infected, ic_initial_activated, mapping, steps=10, gamma=0.1, beta=0.1, behavioral_change_parameter = 3):
    """
    Runs the SIR simulation with influence from IC model.
    
    Parameters:
        graph (nx.Graph): The graph representing the epidemic network (SIR).
        communication_graph (nx.Graph): The communication network (IC).
        initial_infected (set): The initial set of infected nodes.
        ic_initial_activated (set): The initial set of activated nodes in the communication network (IC).
        steps (int): Number of time steps to run the simulation.
        gamma (float): Recovery rate in the SIR model.
        
    Returns:
        dict: Record of infected nodes at each time step.
        dict: Record of recovered nodes at each time step.
    """
    infected = []
    
    for node in initial_infected:
        infected.append(node)
    infected = set(infected)

    recovered = set()

    beta_values = {node: beta for node in graph.nodes()}
    
    neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes()}

    for step in range(steps):
        
        # Get the newly infected and recovered nodes for this step
        new_infected, new_recovered = sir_model_step(
            graph, infected, beta_values, recovered, ic_initial_activated, beta, neighbors, mapping, gamma, behavioral_change_parameter
        )
        
        # Update the infected and recovered sets
        infected.update(new_infected)
        infected.difference_update(new_recovered)
        recovered.update(new_recovered)
        
        # print('infected: ', infected)
        # print('recovered: ', recovered)
        # print('===============')

        # Now update the IC model, activating new nodes if applicable
        ic_activated_nodes = set()  # Reset activated nodes for the IC model each step
        for node in ic_initial_activated:
            # Assuming IC model activates neighbors based on activation probability
            for neighbor in communication_graph.neighbors(node):
                if random.random() < communication_graph[node][neighbor]['weight']:  # Probability of activation
                    ic_activated_nodes.add(neighbor)
        
        # Update the activated nodes set
        ic_initial_activated.update(ic_activated_nodes)

    return infected, recovered

def greedy_blocking_strategy(G_sir, G_ic, initial_infected, initial_activated,
                              candidate_nodes, cost_dict, budget, mapping, beta=0.1, gamma=0.1, T=10, simulations=10, behavioral_change_parameter=3):
    """
    Greedy simulation-based blocking strategy.
    
    Args:
        G_sir (nx.Graph): Contact network (epidemic).
        G_ic (nx.Graph): Communication network (rumor).
        beta_dict (dict): Node-wise infection rates.
        initial_infected (set): Initial SIR infected nodes.
        initial_activated (set): Initial IC activated nodes.
        candidate_nodes (set): Nodes to consider for blocking.
        cost_dict (dict): Node → cost (0 to 1).
        budget (float): Budget (e.g., 1.0).
        T (int): Time steps.
        simulations (int): Averaging simulations per evaluation.

    Returns:
        set: Selected nodes to block.
    """
    selected = set()
    remaining_budget = budget

    def avg_spread(blocked_nodes):
        total = 0
        for _ in range(simulations):
            G_sir_copy = G_sir.copy()
            G_ic_copy = G_ic.copy()
            G_sir_copy.remove_nodes_from(blocked_nodes)
            G_ic_copy.remove_nodes_from(blocked_nodes)
            inf = initial_infected - blocked_nodes
            act = initial_activated - blocked_nodes
            infected, recovered = run_joint_diffusion(
                G_sir_copy, G_ic_copy, inf, act, mapping, T, gamma, beta, behavioral_change_parameter
            )
            result = len(infected) + len(recovered)
            total += result
        return total / simulations

    current_spread = avg_spread(set())

    while True:
        best_node = None
        best_gain = 0
        # print('budget: ', remaining_budget)

        for node in candidate_nodes - selected:
            if cost_dict[node] > remaining_budget:
                continue

            blocked = selected | {node}
            spread = avg_spread(blocked)
            gain = current_spread - spread
            gain_per_cost = gain / cost_dict[node]

            if gain_per_cost > best_gain:
                best_gain = gain_per_cost
                best_node = node

        if best_node is None:
            break

        selected.add(best_node)
        remaining_budget = round(remaining_budget - cost_dict[best_node], 2)
        current_spread -= gain

    return selected

def assign_label(selected, G_sir):
    
    for item in selected:
        
        if item in G_sir:
            continue
        else:
            return 'Not Local'
    
    return "Local"

# Helper functions
def load_instance(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def edge_list_to_adj_matrix(edge_list, starting, end):
    G = nx.Graph()
    G.add_nodes_from(range(starting, end))
    G.add_weighted_edges_from(edge_list)
    return nx.to_numpy_array(G)