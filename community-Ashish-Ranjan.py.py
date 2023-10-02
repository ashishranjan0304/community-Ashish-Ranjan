import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random



def import_facebook_data(path):
    data = pd.read_csv(path, sep=" ", names=['src', 'target'])
    adj_list = dict()
    n = 0 # id of last node or total number of nodes minus 1
    for i in range(len(data)):
        s, t = data.iloc[i][0], data.iloc[i][1]
        n = max(s, t, n)
        if s not in adj_list.keys():
            adj_list[s] = set({t})
        else:
            adj_list[s].add(t)
    
    nodes_connectivity_list_fb = []
    for src in adj_list.keys():
        for target in adj_list[src]:
            if src != target:
                nodes_connectivity_list_fb.append([src, target])

    for i in range(n+1):
        nodes_connectivity_list_fb.append([i, i])
    
    return np.array(nodes_connectivity_list_fb)


def import_bitcoin_data(path):
    data = pd.read_csv(path, usecols=[0,1], names=['src', 'target'])
    adj_list = dict()
    n = 0 # id of last node or total number of nodes minus 1
    for i in range(len(data)):
        s, t = data.iloc[i][0], data.iloc[i][1]
        n = max(s, t, n)
        if s not in adj_list.keys():
            adj_list[s] = set({t})
        else:
            adj_list[s].add(t)
    
    nodes_connectivity_list_btc = []
    for src in adj_list.keys():
        for target in adj_list[src]:
            if src != target:
                nodes_connectivity_list_btc.append([src, target])
    
    # adding self loop edges to the edge_list, 
    # because in the end for both the algorithms it doesn't make a difference (refer report),
    # and the disconnected nodes also get accounted for at the same time
    for i in range(n+1):
        nodes_connectivity_list_btc.append([i, i])

    return np.array(nodes_connectivity_list_btc)


def spectralDecomp_OneIter(edge_array):
    # getting unique node-ids (sorted) in graph formed by edge_array #
    nodes = set()
    for i in range(edge_array.shape[0]):
        nodes.add(edge_array[i][0])
        nodes.add(edge_array[i][1])
    n = len(nodes)
    
    # creating one to one map of node-id to corresponding adjecency matrix index and vice versa #
    node_index = {}
    index_node = {}
    i = 0
    for node_id in nodes:
        node_index[node_id] = i
        index_node[i] = node_id
        i += 1

    # creating adjecency matrix A #
    A = np.zeros([n, n]).astype('int32')
    for i in range(edge_array.shape[0]):
        src = edge_array[i, 0]
        target = edge_array[i, 1]
        
        if src == target:
            continue
        
        src_idx, target_idx = node_index[src], node_index[target]
        A[src_idx, target_idx], A[target_idx, src_idx] = 1, 1
            
    # calculcating laplacian L #
    D = np.zeros([n, n])
    for i in range(n):
        D[i, i] = np.sum(A[i])
    L = D - A

    # getting eigen values and corresponding vectors #
    eVals, eVecs = np.linalg.eigh(L)
    # l, v = scipy.sparse.linalg.eigsh(L, k=4038, which='SA')

    # sorting eigen v's #
    idx = eVals.argsort()
    eVals = eVals[idx]
    eVecs = eVecs[idx]

    # getting fiedler vector (eig vec corresponding to smallest 'close to' non-zero eig val) #
    i = 0
    threshold = 0.9
    while((abs(eVals[i+1] - eVals[i])/eVals[i+1]) < threshold):
        i += 1
    fiedlerVec = eVecs[:, i+1]

    # partitioning nodes into +1 and -1 commnunities using fiedler vector entries #
    # also tracking lowest node id for both communities #    
    lowest_positive = -1
    lowest_negative = -1
    partition = np.array([[index_node[x], -1] for x in range(n)])
    for i in range(n):
        if fiedlerVec[i] >= 0:
            if lowest_positive == -1:
                lowest_positive = index_node[i]
            partition[i][1] = 1
        else:
            if lowest_negative == -1:
                lowest_negative = index_node[i]
    
    # creating graph partition array #
    for i in range(n):
        if partition[i][1] == -1:
            partition[i][1] = lowest_negative
        else:
            partition[i][1] = lowest_positive


    
    # if partition not good, what to return as stopping condition?
    return fiedlerVec, A, np.array(partition)


def spectralDecomposition(edge_array):
    # getting binary partition and checking if partition good # ---------------------------------------- #
    fielder_vec, adj_mat, graph_partition = spectralDecomp_OneIter(edge_array)
    
    community_ids = list(set(graph_partition[:, 1]))

    node_community = dict()     # mapping nodes to communities using dictionary
    for i in range(graph_partition.shape[0]):
        n, c = graph_partition[i, 0], graph_partition[i, 1]
        node_community[n] = c

    # classifying edges according to their respective communities and also getting CUT(community0, community1) #
    edge_array_0 = []
    edge_array_1 = []
    cut = 0
    for i in range(edge_array.shape[0]):
        s, t = edge_array[i, 0], edge_array[i, 1]
        s_comm, t_comm = node_community[s], node_community[t]
        if s_comm != t_comm:   # s and t in different sets -> cut edge
            cut += 1
        elif s_comm == community_ids[0] and t_comm == community_ids[0]: # s and t both in community 0
            edge_array_0.append([s, t])
        else:   # s and t both in community 1
            edge_array_1.append([s, t])
    edge_array_0 = np.array(edge_array_0)
    edge_array_1 = np.array(edge_array_1)

    # getting Vol(comm0) and Vol(comm1) #
    # note: adj_mat index i refers to same node as graph_partition index i
    vol0 = 0
    vol1 = 0
    for i in range(graph_partition.shape[0]):
        degree = np.sum(adj_mat[i, :])
        nc = graph_partition[i, 1]
        if nc == community_ids[0]:
            vol0 += 1
        else:
            vol1 += 1

    # if partition not good, no further spectral decomposition of both parts # --------------------------- #
    # i.e. return partition with all nodes with same community id #
    if vol0 == 0 or vol1 == 0:
        graph_partition[:, 1] = np.ones(graph_partition.shape[0]) * np.min(graph_partition[:, 1])
        return graph_partition
        
    conductance = cut/min(vol0, vol1)
    threshold = 0.3
    if conductance > threshold:
        graph_partition[:, 1] = np.ones(graph_partition.shape[0]) * np.min(graph_partition[:, 1])
        return graph_partition

    # else further partitioning graph into two sub-graphs and getting further communities using spectral decomposition # ---- #
    graph_partition0 = spectralDecomposition(edge_array_0)
    graph_partition1 = spectralDecomposition(edge_array_1)

    # concatenate both partition arrays and return spectral decomposition result # ---------------------------------------- #
    i = 0
    j = 0
    graph_partition = []
    while(i < graph_partition0.shape[0] or j < graph_partition1.shape[0]):
        if i == graph_partition0.shape[0]:
            graph_partition.append(graph_partition1[j])
            j += 1
            continue
        
        if j == graph_partition1.shape[0]:
            graph_partition.append(graph_partition0[i])
            i += 1
            continue

        if graph_partition0[i, 0] < graph_partition1[j, 0]:
            graph_partition.append(graph_partition0[i])
            i += 1
        else:
            graph_partition.append(graph_partition1[j])
            j += 1

    # depth -= 1
    return np.array(graph_partition)


def louvain_one_iter(edge_array):
    nodes = set()
    for edge in edge_array:
        nodes.add(edge[0])
        nodes.add(edge[1])
    n = max(nodes) + 1      # n number of nodes -> 0, 1, . . , n-1
    nodes = set(list(range(n))) 

    # creating adjecency list representation of graph
    adj_list = [-1] * n # -1 signifies node has no edges 
    for edge in edge_array:
        s, t = edge[0], edge[1]
        if s == t:
            continue 
        
        if adj_list[s] == -1:
            adj_list[s] = [t]
        else:
            adj_list[s].append(t)

        if adj_list[t] == -1:
            adj_list[t] = [s]
        else:
            adj_list[t].append(s)

    # calculating adjecency matrix representation of graph
    adj_mat = np.zeros([n, n]).astype('int32')
    for edge in edge_array:
        s, t = edge[0], edge[1]
        if s == t:
            continue
        adj_mat[s, t], adj_mat[t, s] = 1, 1

    m = np.sum(adj_mat)/2

    # getting degree of each node
    node_degree = np.sum(adj_mat, axis=1)

    community_degree = {}    # to keep track total degree in each partition/community
    partition = {}      # will indicate that node i belongs to community j (mapping node-id to its community-id)
    # initially there are n communities as each node is an individual community
    for node in nodes:
        community_degree[node] = node_degree[node]
        partition[node] = node 

    delta_Q_max = -1
    transfer_node = 0
    transfer_from = 0
    transfer_to = 0
    while(delta_Q_max != 0):
        # print(delta_Q_max)
        partition[transfer_node] = transfer_to
        dg= node_degree[transfer_node]
        community_degree[transfer_from] -= dg
        community_degree[transfer_to] += dg

        delta_Q_max = 0
        for node in nodes:
            cc = partition[node] # current community
            # getting number of edges of current node in all its neighbour communities respectively 
            # (number of edges in current community of node also calculated together)
            if adj_list[node] == -1:
                continue
            else:
                node_edges = adj_list[node]
                num_edges_in_nc = {} # nc -> neighbour community
                for adj_node in node_edges:
                    adj_node_comm = partition[adj_node]
                    if adj_node_comm not in num_edges_in_nc.keys():
                        num_edges_in_nc[adj_node_comm] = 1
                    else:
                        num_edges_in_nc[adj_node_comm] += 1
                if cc not in num_edges_in_nc.keys():
                    num_edges_in_nc[cc] = 0
            
            # calculating delta_Q for all possible transfers of current node
            e_cc = num_edges_in_nc[cc] # number edges of current node in its current community
            D_cc = community_degree[cc] # total degree of nodes current community
            d = node_degree[node] # degree of current node
            del num_edges_in_nc[partition[node]]
            for nc in num_edges_in_nc.keys():
                e_nc = num_edges_in_nc[nc] # number edges of current node in its neighbour community
                D_nc = community_degree[nc] # total degree of nodes neighbour community
                delta_Q = (e_nc-e_cc)/m  +  (D_cc**2 + D_nc**2 - (D_cc - d)**2 - (D_nc +d)**2)/((2*m)**2)
                
                # storing the max of all possible node transfers
                if delta_Q > delta_Q_max:
                    delta_Q_max = delta_Q
                    transfer_node = node
                    transfer_from = cc
                    transfer_to = nc
    
    # converting partition dictionary to partition np array with proper community_id format #
    communities = {}
    for node in partition.keys():
        comm = partition[node]
        if comm not in communities.keys():
            communities[comm] = [node]
        else:
            communities[comm].append(node)
    
    partition = []
    for c in communities.keys():
        c_id = min(communities[c])
        for node in communities[c]:
            partition.append([node, c_id])
    
    return np.array(partition)


def createSortedAdjMat(partition, edge_array):
    nodes = set()
    for i in range(edge_array.shape[0]):
        nodes.add(edge_array[i][0])
        nodes.add(edge_array[i][1])
    n = max(nodes) + 1

    adj_mat = np.zeros([n, n]).astype('int32')
    for i in range(edge_array.shape[0]):
        src = edge_array[i, 0]
        target = edge_array[i, 1]
        
        if src == target:
            continue

        adj_mat[src, target], adj_mat[target, src] = 1, 1

    # displaying sorted adjecency matrix #

    sorted_graph_partition = partition[partition[:, 1].argsort()]
    sorted_idx = sorted_graph_partition[:, 0]

    sorted_adj_matrix = adj_mat[sorted_idx]
    sorted_adj_matrix = (sorted_adj_matrix.T)[sorted_idx]

    print("------- Displaying sorted adjecency matrix -------")
    plt.figure()
    plt.imshow(sorted_adj_matrix)
    plt.colorbar()
    plt.show()

    # creating graph # 
    #G = nx.from_numpy_matrix(adj_mat)
    G = nx.Graph()

    # Add nodes to the graph (assuming nodes are labeled 0, 1, 2, ...)
    num_nodes = adj_mat.shape[0]
    G.add_nodes_from(range(num_nodes))

    # Add edges to the graph based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(i, j)

    # displaying original graph for reference #
    print("------- Displaying original graph for reference -------")
    plt.figure()
    nx.draw(G, node_size=10)
    plt.show()

    # visualizing communities by first getting color code #
    # below code is to get a 'color_list' in order to color different communities in graph # --- #

    # getting set of communities (that are identified by lowest index of that community) #
    communities = set()
    for i in range(partition.shape[0]):
        communities.add(partition[i, 1])

    # getting nc number of random rgb codes where nc is number of communities #
    nc = len(communities)
    intensity = [i for i in range(256)]
    rgb_codes = zip([random.choice(intensity) for _ in range(nc)], \
                    [random.choice(intensity) for _ in range(nc)], \
                    [random.choice(intensity) for _ in range(nc)])
    colors = [(r/255, g/255, b/255) for r, g, b in rgb_codes]

    # mapping one color to each community in a dictionary #
    community_color = dict()
    i = 0
    for c in communities:
        if c not in community_color.keys():
            community_color[c] = colors[i]
            i += 1

    # getting color map list -> color for each node in graph using above community-color dictionary #
    color_map = []
    for i in range(partition.shape[0]):
        c = partition[i, 1] # community to which the node belongs
        color_map.append(community_color[c])

    # finally got color list using above code # ------------------------------------------------ #

    # displaying graph of communities #
    print("------- Displaying color coded community graph -------")
    plt.figure()
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_color=color_map, node_size=10)
    # nx.draw(G, with_labels=True, node_color=color_map, node_size=100)
    plt.show()

    return sorted_adj_matrix

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data(r"C:\Users\A.Ranjan3\Downloads\facebook_combined (1).txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)


    # Sorting the Fiedler vector
    sorted_fiedlerVec = np.sort(fielder_vec_fb)

    # Plotting the sorted Fiedler vector, adjacency matrix, and graph partition #
    plt.figure(figsize=(12, 4))

    # Plot the sorted Fiedler vector
    plt.subplot(131)
    plt.plot(sorted_fiedlerVec)
    plt.title('Sorted Fiedler Vector')

    # Plot the adjacency matrix
    plt.subplot(132)
    plt.imshow(adj_mat_fb, cmap='binary', interpolation='none')
    plt.title('Adjacency Matrix')

    # Plot the graph partition
    plt.subplot(133)
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_fb)
    pos = nx.spring_layout(G)
    node_colors = [partition[1] for partition in graph_partition_fb]
    nx.draw(G, pos=pos, node_color=node_colors, cmap=plt.cm.viridis)
    plt.title('Graph Partition')

    plt.show()


    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)
    plt.figure(figsize=(6, 6))
    plt.imshow(clustered_adj_mat_fb, cmap='binary', interpolation='none')
    plt.title('Sorted Adjacency Matrix')
    plt.show()

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_fb)
    pos = nx.spring_layout(G)
    node_colors_louvain = [partition[1] for partition in graph_partition_louvain_fb]
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=pos, node_color=node_colors_louvain, cmap=plt.cm.viridis)
    plt.title('Louvain Communities (One Iteration)')
    plt.show()


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data(r"C:\Users\A.Ranjan3\Downloads\soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)
        # Sorting the Fiedler vector
    sorted_fiedlerVec = np.sort(fielder_vec_btc)

    # Plotting the sorted Fiedler vector, adjacency matrix, and graph partition #
    plt.figure(figsize=(12, 4))

    # Plot the sorted Fiedler vector
    plt.subplot(131)
    plt.plot(sorted_fiedlerVec)
    plt.title('Sorted Fiedler Vector btc')

    # Plot the adjacency matrix
    plt.subplot(132)
    plt.imshow(adj_mat_btc, cmap='binary', interpolation='none')
    plt.title('Adjacency Matrix btc')

    # Plot the graph partition
    plt.subplot(133)
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_btc)
    pos = nx.spring_layout(G)
    node_colors = [partition[1] for partition in graph_partition_btc]
    nx.draw(G, pos=pos, node_color=node_colors, cmap=plt.cm.viridis)
    plt.title('Graph Partition btc')

    plt.show()

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)
    plt.figure(figsize=(6, 6))
    plt.imshow(clustered_adj_mat_btc, cmap='binary', interpolation='none')
    plt.title('Sorted Adjacency Matrix btc')
    plt.show()

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    G = nx.Graph()
    G.add_edges_from(nodes_connectivity_list_btc)
    pos = nx.spring_layout(G)
    node_colors_louvain = [partition[1] for partition in graph_partition_louvain_btc]
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=pos, node_color=node_colors_louvain, cmap=plt.cm.viridis)
    plt.title('Louvain Communities (One Iteration) btc')
    plt.show()