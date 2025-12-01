import networkx as nx
from util import *
import numpy as np
import random
import json
import pickle

# for iter in range(100):

# Iterate through multiple initial settings

for iter in range(1000):

    N = random.randint(200, 300)
    social_rate = random.uniform(0.5, 0.8)
    M = int(N * social_rate)
    print('N: ', N)
    print('M: ', M)
    
    # Step 1: Define the Networks with Partial Overlap
    communication_nodes = range(N, N+M)  # Nodes for the communication network
    contact_nodes = range(0, N)  # Nodes for the contact network

    initial_infected = set(random.sample(contact_nodes, 3))
    initial_activated = set(random.sample(communication_nodes, 3))

    budget = 1
    step = 100

    for beta in [0.1, 0.01, 0.001]:
        for gamma in [0.1, 0.01, 0.001]:
            for behavioral_change_parameter in [3, 10, 100]:
                # Step 1: Define the Networks with Partial Overlap

                # Create the networks
                communication_network = nx.connected_watts_strogatz_graph(M, random.randint(5,11), 0.2)
                contact_network = nx.connected_watts_strogatz_graph(N, random.randint(2,6), 0.2)
                
                communication_network = nx.DiGraph(communication_network)
                contact_network = nx.DiGraph(contact_network)
                
                for a, b in communication_network.edges():
                    weight = random.randrange(40, 100)
                    weight = round(weight/100, 2)
                    communication_network[a][b]['weight'] = weight


                # Relabel nodes to reflect separate node sets
                communication_network = nx.relabel_nodes(communication_network, lambda x: communication_nodes[x])
                contact_network = nx.relabel_nodes(contact_network, lambda x: contact_nodes[x])

                contact_nodes_set = set(contact_network.nodes())
                communication_nodes_set = set(communication_network.nodes())
                all_nodes = contact_nodes_set.union(communication_nodes_set)
                all_nodes.difference_update(initial_infected)

                cost_dict = {}
                for node in contact_nodes_set:
                    cost_dict[node] = 0.3
                for node in communication_nodes_set:
                    cost_dict[node] = 0.2
    
                # add interplay layer in tensor
                left_in_contact = list(range(0, N))
                left_in_communication = list(range(N, N+M))
                
                mapping = {}
                
                for user in communication_nodes_set:
                    
                    human = random.choice(left_in_contact)
                    mapping[user] = human
                    left_in_contact.remove(human)

        
                original_infected, original_recovered = run_joint_diffusion(contact_network, communication_network, 
                                                    initial_infected, initial_activated, mapping, steps=step, gamma=gamma, beta=beta, behavioral_change_parameter = behavioral_change_parameter)
                print('infected: ', len(original_infected))
                # print(original_infected)
                print('recovered: ', len(original_recovered))
                # print(original_recovered)
                        
                selected = greedy_blocking_strategy(contact_network, communication_network, initial_infected, initial_activated,
                                            all_nodes, cost_dict, budget, mapping, beta, gamma, T=step, simulations=5, behavioral_change_parameter=behavioral_change_parameter)

                print(selected)

                label = assign_label(selected, contact_network)
                print(label)
                print('=========')

                results = {}
                # edgelist_contact_backup = list(contact_network.edges())
                # edgelist_contact = []
                # for item in edgelist_contact_backup:
                #     u = item[0]
                #     v = item[1]
                #     item = (u, v, beta)
                # results['contact_network'] = list(contact_network.edges())
                # edgelist_commu_backup = list(communication_network.edges())
                # edgelist_commu = []
                # for item in edgelist_commu_backup:
                #     u = item[0]
                #     v = item[1]
                #     item = (u, v, communication_network[u][v]['weight'])
                #     edgelist_commu.append(item)
                edgelist_contact = []
                for u, v in contact_network.edges():
                    edgelist_contact.append((u, v, beta))
                edgelist_commu = []
                for u, v in communication_network.edges():
                    edgelist_commu.append((u, v, communication_network[u][v]['weight']))
                results['communication_network'] = edgelist_commu
                results['contact_network'] = edgelist_contact
                # results['contact_network'] = nx.to_numpy_array(contact_network).tolist()
                # results['communication_network'] = nx.to_numpy_array(communication_network).tolist()
                results['initial_infected'] = initial_infected
                results['initial_activated'] = initial_activated
                results['mapping'] = mapping
                results['beta'] = beta
                results['gamma'] = gamma
                results['behavioral_change_parameter'] = behavioral_change_parameter
                results['contact_cost'] = 0.3
                results['communication_cost'] = 0.2
                results['selected'] = selected
                results['label'] = label

                # # Save the results to a file
                # filename = f"results_iter_{iter}.json"
                # with open(filename, 'w') as f:
                #     json.dump(results, f)

                # Save the results to a pkl file
                filename = f"results2_iter_{iter}_beta_{beta}_gamma_{gamma}_bp_{behavioral_change_parameter}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(results, f)