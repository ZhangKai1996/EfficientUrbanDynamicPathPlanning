import heapq

from env.geo import base_speed


def dijkstra_path(graph, start_node, end_node, matrix=None, weight_key='dynamic_weight'):
    # 最小堆，存储 (累计成本, 当前节点, 路径)
    heap = [(0, start_node, [start_node])]
    visited = set()

    while heap:
        cost, current, path = heapq.heappop(heap)
        if current == end_node: return cost, path
        if current in visited: continue
        visited.add(current)

        for neighbor in graph[current].keys():
            if neighbor in visited: continue

            if matrix is None:
                weight = graph[current][neighbor][0][weight_key]
            else:
                length = graph[current][neighbor][0]['length']
                times = matrix[(current, neighbor)]
                weight = length / (base_speed * times)
                # weight = matrix[f'{current}-{neighbor}']
            heapq.heappush(heap, (cost + weight, neighbor, path + [neighbor]))
    return float('inf'), []  # 如果没有路径


def yen_m_shortest_paths(graph_, source, target, k, weight_key='dynamic_weight'):
    graph = graph_.copy()
    cost, path = dijkstra_path(graph, source, target, weight_key=weight_key)
    if not path: return [], []

    paths, costs = [path, ], [cost, ]
    for j in range(k):
        print(j, round(cost, 3), path)
        for i, u in enumerate(path[:-1]):
            v = path[i + 1]
            print('\t', u, v, list(graph[u].keys()), list(graph[v].keys()), end=' | ')

            if i == 0 and len(graph[u]) <= 1: continue
            if len(graph[u]) <= 2: continue
            if len(graph[v]) <= 2: continue

            graph.remove_edge(u, v)
            print(list(graph[u].keys()), list(graph[v].keys()))
            break

        c2, p2 = dijkstra_path(graph, source, target, weight_key=weight_key)
        done = len(p2) > 0 and p2[0] == source and p2[-1] == target
        print(done, c2, p2)
        path, cost = p2[:], c2
        paths.append(p2)
        costs.append(c2)
    return paths, costs

