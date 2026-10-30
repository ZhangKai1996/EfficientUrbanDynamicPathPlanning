import math

import numpy as np
from tqdm import tqdm

from algo.baselines import dijkstra_path
from env.utils import haversine
from .rendering import StaticRender


class DPPEnvironment:
    def __init__(self, graph, p, num_action,
                 e=100.0,
                 zeta=0.0,
                 b=0.0,
                 a=-1.0,
                 c=0.0,
                 alpha=1.0,
                 gamma=0.99,
                 dfr_generator=None,
                 od_pairs=None):

        print('-------Environment Parameters--------')
        print('\t e =', e)
        print('\t zeta =', zeta)
        print('\t b =', b)
        print('\t a =', a)
        print('\t c =', c)
        print('\t alpha =', alpha)
        print('\t gamma =', gamma)
        print('\t Number of OD pairs:', len(od_pairs))

        self.p = p
        self.graph = graph
        self.num_action = num_action
        self.nodes: list = list(graph.nodes)

        self.od_pairs = od_pairs
        self.start_node_idx = None
        self.end_node_idx = None

        self.cur_node_idx = None
        self.last_node_idx = None
        self.dist = {}

        self.e = e
        self.b = b
        self.a = a
        self.c = c
        self.zeta = zeta
        self.alpha = alpha
        self.gamma = gamma

        self.dfr = dfr_generator
        self.cv_render = None

    @property
    def start_node(self): return self.nodes[self.start_node_idx]

    @property
    def end_node(self): return self.nodes[self.end_node_idx]

    @property
    def cur_node(self): return self.nodes[self.cur_node_idx]

    @property
    def last_node(self): return self.nodes[self.last_node_idx]

    def reset(self, reuse=False, render=None):
        if self.start_node_idx is None: reuse = False

        if not reuse:
            od_pair = self.od_pairs.pop(0)
            self.od_pairs.append(od_pair)
            self.start_node_idx = self.nodes.index(od_pair[0])
            self.end_node_idx = self.nodes.index(od_pair[1])

            # if self.end_node not in self.dist.keys():
            #     dist = {}
            #     for node in tqdm(self.nodes, desc='Finding shortest paths ...'):
            #         cost, _ = dijkstra_path(self.graph, node, self.end_node, weight_key='length')
            #         dist[node] = cost
            #     self.dist[self.end_node] = dist

            self.dfr.update()

        self.cur_node_idx = self.start_node_idx
        if render and self.cv_render is None:
            self.cv_render = StaticRender(self.graph)
        return self.get_state()

    def get_state(self):
        next_nodes = self.p[self.cur_node]
        node_info = [self.cur_node_idx, self.end_node_idx]
        dyn = self.dfr.get_dynamics(self.cur_node)
        mask = np.zeros((self.num_action,))
        mask[len(next_nodes):] = -np.inf
        return node_info, dyn, mask

    def calculate_potential(self, u, v, verbose=False):
        cost = self.dfr.get_cost(u, v) * self.alpha

        nodes = self.graph.nodes
        d1 = haversine(nodes[u], nodes[self.end_node])
        d2 = haversine(nodes[v], nodes[self.end_node])

        # d1 = self.dist[self.end_node][u]
        # d2 = self.dist[self.end_node][v]

        phi_s = -d1 * self.zeta + self.b
        phi_s_prime = -d2 * self.zeta + self.b
        diff = 1.0 * phi_s_prime - phi_s

        if self.a >= 0.0: diff = min(self.a, diff)
        if self.c < 0.0: diff = max(self.c, diff)

        reward = -cost + diff
        if verbose:
            print('Cost:{:>7.4f}'.format(cost), end=' ==> ')
            print('Phi1:{:>7.4f}, Phi2:{:>7.4f}'.format(phi_s, phi_s_prime), end=', ')
            print('Diff:{:>7.4f}'.format(diff), end=', ')
            print('Reward:{:>7.4f}'.format(reward), end=' ==> ')
        return reward, cost

    def step(self, action, verbose=False):
        cur_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[cur_node]
        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.end_node_idx)

        # Reward (COST)
        shaped_reward, cost = self.calculate_potential(cur_node, next_node, verbose=verbose)
        original_reward = -cost
        if done:
            shaped_reward += self.e
            original_reward += self.e

        self.last_node_idx = self.cur_node_idx
        self.cur_node_idx = next_node_idx

        # Dynamics update per step
        # self.dfr.update()
        return self.get_state(), [shaped_reward, original_reward], done, [cost, ]

    def render(self, *args, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(*args, **kwargs)

    def close(self):
        if self.cv_render is not None:
            self.cv_render.close()
            self.cv_render = None
