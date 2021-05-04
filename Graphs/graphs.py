import numpy as np
import random

import networkx as nx
import matplotlib.pyplot as plt

def ER(N=32, P=0.25):
    edges = []
    # Randomly add edges
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < P: edges.append((i,j))
    return edges

def BA(N=32, M=4):
    edges = []
    degrees = [0]*M
    # When adding first new node M, there is an edge
    # from all initial nodes to node M
    for i in range(M):
        edges.append((i, M))
        degrees[i] = 1
    degrees.append(M)
    # Add edges for remainig N-M-1 nodes
    for i in range(M+1, N):
        # Idea to use np.random.choice() taken, but modified,
        # from https://github.com/seungwonpark/RandWireNN/blob/master/model/graphs/ba.py
        choice = np.random.choice(range(i), size=M, replace=False, p=degrees/np.sum(degrees))
        for c in choice:
            edges.append((c, i))
            degrees[c] += 1
        degrees.append(M)

    return edges

def WS(N=32, K=8, P=0.75):
    edges = []
    # Connect each node to its K/2 neighbors
    for i in range(N):
        for j in range(i-K//2, i+K//2+1):
            if i == j: continue
            start = min(i, j%N)
            end = max(i, j%N)
            if (start, end) not in edges: edges.append((start, end))
    # Randomly rewire edges
    for i in range(1, K//2+1):
        for j in range(N):
            # If <P, randomly rewire edge (j,j+i) to (j,j+choice)
            if random.random() < P:
                valid_choice = False
                # Search until valid choice is made
                while(not valid_choice):
                    choice = random.randint(1,N)
                    # Make sure (j,j+choice) is not already in the graph
                    if (j, (j+choice)%N) not in edges and j != (j+choice)%N:
                        valid_choice = True
                        # Add new edge
                        start = min(j, (j+choice)%N)
                        end = max(j, (j+choice)%N)
                        edges.append((start, end))
                        # Remove old edge
                        start = min(j, (j+i)%N)
                        end = max(j, (j+i)%N)
                        edges.remove((start, end))
    return edges

# Given a list of edges in directed graph,
# figure out which ones are input or output nodes
def input_output_nodes(edges, N=32):
    inputs, outputs = [], []
    for i in range(N):
        is_input, is_output = True, True
        for edge in edges:
            if edge[0] == i: is_output = False
            if edge[1] == i: is_input = False
        if is_input: inputs.append(i)
        if is_output: outputs.append(i)
    return inputs, outputs

def draw_graph(edges, inputs, outputs):
    G = nx.MultiDiGraph()
    G.add_edges_from(edges)
    color_map = []
    for node in G:
        if node in inputs: color_map.append('blue')
        elif node in outputs: color_map.append('red')
        else: color_map.append("green")
    plt.figure(figsize=(8,8))
    nx.draw(G, node_color=color_map, pos=nx.kamada_kawai_layout(G))
    plt.show()

def save_graph(N, edges, type='WS', id=0):
    f = open("Graphs/SavedGraphs/{}_{}".format(type, id), 'w')
    f.write("{}\n".format(N))
    for edge in edges:
        f.write("{}\n".format(edge))
    f.close()

if __name__=="__main__":
    N = 32
    for id in range(3):
        edges = WS(N, K=4, P=.75)
        save_graph(N, edges, 'WS', id)
        inputs, outputs = input_output_nodes(edges, N)
        draw_graph(edges, inputs, outputs)