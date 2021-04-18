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
                    if (j, (j+choice)%N) not in edges:
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

if __name__=="__main__":
    #print(ER())
    #print(BA())
    print(WS())