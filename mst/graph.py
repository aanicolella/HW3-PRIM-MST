import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # get num vertices (V)
        V = len(self.adj_mat)
        # for tracking if nodes are in MST--initialize to false for number of nodes in graph
        visited = [False] * V
        # initialize MST node tracking
        buildMST = np.zeros(self.adj_mat.shape)
        # for tracking MST weight
        MST_weight= 0
        
        # set start node, using 0 in this case, but could be adjusted to random node
        start = 0
        visited[start] = True
        # initialize and build pq for start node
        pq = [(w, start, v) for v, w in enumerate(self.adj_mat[start]) if w > 0]
        heapq.heapify(pq)

        # now that everything's initialized, prims algorithm for finding MST
        while pq:
            # pop values and check if vertex already visited--nodes may have multiple key values in pq, first=lowest, ignore rest
            w, u, v = heapq.heappop(pq)
            if visited[v]:
                continue
            visited[v] = True
            # add to MST adj matrix and update MST weight tracking
            buildMST[v,u], buildMST[u,v] = w, w
            MST_weight += w
            # checking for and adding unvisited neighbors to pq
            for neighbor, w in enumerate(self.adj_mat[v]):
                if w > 0 and not visited[neighbor]:
                    heapq.heappush(pq, (w,v,neighbor))
        # save mst to proper slot in self
        self.mst = buildMST