import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import ndlib.models.opinions as op
import numpy as np
import random

random.seed(42)


# Step 1: Define the Networks with Partial Overlap
communication_nodes = range(100, 200)  # Nodes for the communication network
contact_nodes = range(0, 100)  # Nodes for the contact network

budget = 0
beta = 0

# Create the networks
communication_network = nx.erdos_renyi_graph(n=100, p=0.2)
contact_network = nx.erdos_renyi_graph(n=100, p=0.05)

# Relabel nodes to reflect separate node sets
communication_network = nx.relabel_nodes(communication_network, lambda x: communication_nodes[x])
contact_network = nx.relabel_nodes(contact_network, lambda x: contact_nodes[x])

# Step 2: Define the Interplay
# Create a tensor to store the structure of both networks and their interplay
# Dimensions: (total number of nodes, total number of nodes, 2)
total_nodes = len(communication_nodes) + len(contact_nodes)
tensor = np.zeros((total_nodes, total_nodes, 3))

# add intra network layers in tensor
for i in communication_nodes:
    tensor[i, i, 0] = 1
for i in contact_nodes:
    tensor[i, i, 1] = 1
    

for i,j in contact_network.edges():
    tensor[i, j, 1] = beta
    tensor[j, i, 1] = beta
    contact_network[i][j]['beta'] = beta
    contact_network[j][i]['beta'] = beta
    
# add interplay layer in tensor
left_in_contact = list(range(0, 100))
left_in_communication = list(range(100, 200))

for i in left_in_contact[:]:
    left_in_contact.remove(i)
    random = np.random.uniform(0, 1)
    if random < 0.8:
        user_id = np.random.choice(left_in_communication)
        tensor[i, user_id, 2] = 1
        tensor[user_id, i, 2] = 1
        left_in_communication.remove(user_id)
        
print(np.where(tensor[100, :, 2] == 1)[0])
print(np.where(tensor[0, :, 2] == 1)[0])

# Step 3: Initialize the Models
     
sir_model = ep.SIRModel(contact_network)
ic_model = ep.IndependentCascadesModel(communication_network)

# Configure the models
epi_source = 0
config = mc.Configuration()
config.add_model_initial_configuration('Infected', [epi_source])
config.add_model_parameter('beta', beta)
config.add_model_parameter('gamma', 0)  # Recovery rate for SIR
# sir_model.set_edge_parameters('beta')
sir_model.set_initial_status(config)

rumor_source = 100
config_ic = mc.Configuration()

for i,j in communication_network.edges():
    weight = np.random.uniform(0.3, 0.8)
    tensor[i, j, 0] = weight
    tensor[j, i, 0] = weight
    config.add_edge_configuration('threshold', (i,j), weight)
    config.add_edge_configuration('threshold', (j,i), weight)
    
config_ic.add_model_initial_configuration('Infected', [rumor_source])
# config_ic.add_model_parameter('fraction_infected', 0.1)  # Initial infected fraction for IC
ic_model.set_initial_status(config_ic)

# # Initialize the tensor with the network structures
# for i, j in communication_network.edges():
#     weight = np.random.uniform(0, 0.5)
#     tensor[i, j, 0] = weight  # Communication network

# for i, j in contact_network.edges():
#     tensor[i, j, 1] = beta  # Contact network

# Step 4: Simulate the Diffusion Process
for t in range(100):  # Simulate for 100 time steps
    # Simulate one step in the communication network and the contact network
    ic_model.iteration()


    # Get the current status of the communication network
    ic_status = ic_model.status

    # Update the contact network based on the communication network
    for node in communication_network.nodes():
        if ic_status[node] == 1:  # If the node is activated
            if (tensor[node, :, 2] == 1).any():
                human_id = np.where(tensor[node, :, 2] == 1)[0][0]
                # Increase activation probability in the contact network
                for neighbor in contact_network.neighbors(human_id):
                    tensor[human_id, neighbor, 1] = 0.1
                    tensor[neighbor, human_id, 1] = 0.1
                    # contact_network[human_id][neighbor]['beta'] = beta*10
                    # contact_network[neighbor][human_id]['beta'] = beta*10
                    
    
    sir_model.iteration()
                    
    # # capture information from sir model for next iteration
    # node_statuses = sir_model.status
    
    # # update the contact network configuration for next iteration 
    # config = mc.Configuration()
    

    # Update the tensor to reflect the current state
    # This can include updating the activation probabilities or other parameters

# Step 5: Collect and analyze the results

sir_node_statuses = sir_model.status

s_count = list(sir_node_statuses.values()).count(0)
i_count = list(sir_node_statuses.values()).count(1)
r_count = list(sir_node_statuses.values()).count(2)

print(f"Epi:  Susceptible: {s_count}, Infected: {i_count}, Recovered: {r_count}")

ic_node_statuses = ic_model.status
infected_count = list(ic_node_statuses.values()).count(1)
removed = list(ic_node_statuses.values()).count(2)
other = list(ic_node_statuses.values()).count(0)

print(f"Rumor:  Not infected: {other}, Infected: {infected_count}, Infected earlier: {removed}")