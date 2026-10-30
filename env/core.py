import numpy as np

from env.geo import base_speed


class DFRGenerator:
    def __init__(self, graph, neighbors=None, length_limit=None):
        self.neighbors = neighbors
        self.length_limit = length_limit
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.node2id = {node: i for i, node in enumerate(self.nodes)}

        self.edges = []
        self.edge2id = {}
        lengths = []
        for i, (u, v, data) in enumerate(graph.edges(data=True)):
            self.edge2id[(u, v)] = i
            self.edges.append((u, v))
            lengths.append(data.get('length', 1))

        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        self.length = np.fromiter(lengths, dtype=np.float32, count=self.n_edges)

        self.mask = (self.length >= length_limit[0]) & (self.length <= length_limit[1])
        print('\t The number of road with dynamics: ', self.mask.sum())

        edges_dict = neighbors['edges_dict']
        self.hop_indices = {}
        for node, hop_edges in edges_dict.items():
            self.hop_indices[node] = [self.edge2id[edge] for edge in hop_edges]
        self.w = None

    def update(self):
        self.w = np.ones(self.n_edges, dtype=np.float32)
        if self.mask is not None:
            self.w[self.mask] = np.random.uniform(0.01, 1.0, self.mask.sum()).astype(np.float32)

    def get_w(self):
        return {edge: self.w[eid] for edge, eid in self.edge2id.items()}

    def get_dynamics(self, cur_node):
        max_len = self.neighbors['max_len']
        dyn = np.zeros((max_len, )).astype(np.float32)
        if len(self.hop_indices) <= 0: return dyn

        hop_ind = self.hop_indices[cur_node]
        dyn[:len(hop_ind)] = 1.0-self.w[hop_ind]
        return dyn

    def get_cost(self, u, v):
        times = self.w[self.edge2id[(u, v)]]
        length = self.graph[u][v][0]['length']
        cost = length / (base_speed * times)
        return cost
