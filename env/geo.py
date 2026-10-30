import math

import numpy as np
import osmnx as ox
import networkx as nx

from env.utils import haversine
from env.rendering import StaticRender

base_speed = 60000.0 / 3600.0  # m/s


def load_graph(place_name='Nanjing, China',
               network_type='all',
               center_idx=0,
               radius=1e6,
               remove=True,
               render=False,
               **kwargs):
    print('-------Loading urban graph-----------')
    print('\t Place name:', place_name)
    print('\t Network type:', network_type)
    print('\t Radius:', radius)
    print('\t Center index:', center_idx)
    print('\t Remove isolated nodes:', remove)
    print('\t Render the graph:', render)
    print('\t Graph process:')

    graph_ = ox.graph_from_place(place_name, network_type=network_type)
    print(f"\t\t Initial {len(graph_.nodes)} nodes and {len(graph_.edges)} edges")
    print(f"\t\t Strong Connection: {nx.is_strongly_connected(graph_)}")

    graph, center_node = extract_subgraph(radius, graph_, center_idx)
    print(f"\t\t After clipped by distance:", end=' ')
    print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"\t\t Strong Connection: {nx.is_strongly_connected(graph)}")

    if remove:
        g_strong = max(nx.strongly_connected_components(graph), key=len)
        graph = graph.subgraph(g_strong).copy()
        print(f"\t\t After strongly connected:", end=' ')
        print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")

    p, num_action = remove_isolated_nodes(graph)
    print(f"\t\t After removing isolated nodes:", end=' ')
    print(f"{len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print(f"\t\t Number of action: {num_action}")

    print(f"\t\t Center node (origin): ", center_node, center_node in list(graph.nodes))
    center_node_, ranked_nodes = get_rank(graph, center_node, change_end_node=True)
    print(f"\t\t Center node (new): ", center_node_, center_node_ in list(graph.nodes))
    print(f"\t\t Number of ranked nodes: ", len(ranked_nodes))

    add_dynamic_weight(graph)
    print(f"\t\t Strong Connection: {nx.is_strongly_connected(graph)}")
    od_pairs = sector_od_pairs(graph, center_node, n_sectors=50)

    start_node = list(graph.nodes)[ranked_nodes[-1]]
    od_pairs = [(start_node, center_node_), ]
    if render:
        city_name = place_name.split(',')[0]
        markers = {'D': center_node_}
        for i, od_pair in enumerate(od_pairs):
            markers.update({f'O{i}': od_pair[0]})
        name = 'data/{}_{}_{}'.format(city_name, int(radius), center_idx)
        StaticRender(graph).draw(markers=markers, name=name, show=False)
    return graph, p, num_action, od_pairs


def get_rank(graph, end_node, change_end_node=False):
    nodes = list(graph.nodes)
    end_node_idx = nodes.index(end_node)

    ret = {}
    for node_idx in range(len(nodes)):
        if node_idx == end_node_idx: continue

        u = graph.nodes[nodes[node_idx]]
        v = graph.nodes[end_node]
        ret[node_idx] = haversine(u, v)

    sorted_items = sorted(ret.items(), key=lambda x: x[1])
    rank_key = [item[0] for item in sorted_items]

    if change_end_node:
        end_node = nodes[rank_key[-1]]
        return get_rank(graph, end_node, change_end_node=False)
    return end_node, rank_key


def sector_od_pairs(graph, center, n_sectors=4):
    nodes = graph.nodes
    center_info = nodes[center]
    cx, cy = center_info['x'], center_info['y']

    # 1) 计算每个节点相对中心的角度和距离
    node_info = []
    for n in list(nodes):
        n_info = nodes[n]
        dx, dy = n_info['x'] - cx, n_info['y'] - cy
        r = np.hypot(dx, dy)
        theta = np.degrees(np.arctan2(dy, dx)) % 360
        node_info.append((n, theta, r))

    # 2) 分扇区，每扇区取最远点
    sector_angle = 360.0 / n_sectors
    sector_centers = [(s + 0.5) * sector_angle for s in range(n_sectors)]
    delta = 0.5 * sector_angle

    sector_upper = {}  # sector_idx -> (node, r)   最外环(最大值)
    sector_median = {}  # sector_idx -> (node, r)   中间环(中位数)
    for s, center_angle in enumerate(sector_centers):
        cand = []  # 收集该扇区内所有满足条件的 (node, r)
        for n, theta, r in node_info:
            d = abs(theta - center_angle)
            d = min(d, 360 - d)  # 环形最小角差
            if d <= delta: cand.append((n, r))

        if not cand: continue

        cand.sort(key=lambda x: x[1])
        cand_part = [x for x in cand if x[1] >= cand[-1][1] / 2.0]
        sector_median[s] = cand_part[0]
        cand_part = [x for x in cand if x[1] >= cand[-1][1] / 1.0]
        sector_upper[s] = cand_part[0]

    points_median = [sector_median[s][0] for s in sorted(sector_median.keys())]
    od_pairs = [(p, center) for p in points_median]
    points_upper = [sector_upper[s][0] for s in sorted(sector_upper.keys())]
    od_pairs += [(p, center) for p in points_upper]

    # od_pairs = []
    # for i, end_node in enumerate(points_upper):
    #     for j, start_node in enumerate(points_upper):
    #         if i == j: continue
    #         if (j - i) % n_sectors in (1, n_sectors - 1): continue
    #         od_pairs.append((start_node, end_node))

    print(f"\t OD 对: {len(od_pairs)}, 实际连通 OD 对: {len(set(od_pairs))}")
    return od_pairs


def extract_subgraph(radius, graph, center_idx):
    center_node = list(graph.nodes)[center_idx]
    nodes = graph.nodes

    sub_nodes = []
    for i, node in enumerate(list(nodes)):
        dist = haversine(nodes[center_node], nodes[node], km=False)
        if radius < 0 or dist <= radius:
            sub_nodes.append(node)
    subgraph = graph.subgraph(sub_nodes).copy()
    return subgraph, center_node


def remove_isolated_nodes(graph):
    """从图中移除孤立节点：没有连接任何其他节点的节点"""
    while True:
        isolated_nodes = []
        ret, max_number = {}, 0
        for node1 in graph.nodes:
            tmp = []
            for node2, data in graph[node1].items():
                if node1 == node2: continue
                if data.get('length', 1) > 0: tmp.append(node2)

            if len(tmp) <= 0:
                isolated_nodes.append(node1)
            else:
                ret[node1] = tmp
                max_number = max(max_number, len(tmp))

        if len(isolated_nodes) <= 0: break
        graph.remove_nodes_from(isolated_nodes)
    return ret, max_number


def add_dynamic_weight(graph):
    nodes = list(graph.nodes)

    edge_index, edge_attr = [], []
    for u, v, data in graph.edges(data=True):
        times = 1.0
        length = data.get('length', 1)
        traffic_time = length / (base_speed * times)
        data['times'] = times
        data['dynamic_weight'] = traffic_time

        edge_index.append([nodes.index(u), nodes.index(v)])
        edge_attr.append(times)
    return edge_index, edge_attr


def get_hop_neighbors(graph, max_n=3, limit=None, verbose=True):
    if verbose:
        print('------Get ordered n-hop edge neighborhoods--------')
        print('\t max_n =', max_n)

    if max_n == 0: return {'edges_dict': {}, 'max_len': 1}

    nodes = graph.nodes
    edges_dict, max_len, min_len = {}, 0, 1e6
    for u in list(nodes):
        current_layer = {u}

        hop_edges = []
        seen_edges = set()  # 已归档的边
        for n in range(1, max_n + 1):
            next_layer = set()
            for v in current_layer:  # 扩展 n-hop 节点
                for w in set(graph.neighbors(v)):
                    next_layer.add(w)
                    edge = (v, w)
                    if limit is not None:
                        if graph[v][w][0]['length'] < limit[0]: continue
                        if graph[v][w][0]['length'] > limit[1]: continue
                    if edge in seen_edges: continue
                    seen_edges.add(edge)
                    hop_edges.append(edge)
            current_layer = next_layer
        edges_dict[u] = hop_edges
        max_len = max(max_len, len(hop_edges))
        min_len = min(min_len, len(hop_edges))

    if verbose:
        print(f"\t Max edges per layer: {max_len}")
        print(f"\t Min edges per layer: {min_len}")
    return {'edges_dict': edges_dict, 'max_len': max_len}

