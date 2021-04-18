import numpy as np
import random

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

if __name__=="__main__":
    #print(ER())
    print(BA())