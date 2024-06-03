import time

import networkx as nx

from common import GraphLayer
from graph_generator import extract_cluster_subgraph, extract_cluster_list_subgraph


# 0.043622783921728685 0.04541889641851229 no a star
def find_path(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        neighbour_cluster=False,
        paths=None) -> tuple[float, list[int]]:
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    if from_cluster == to_cluster:
        return nx.single_source_dijkstra(extract_cluster_subgraph(layer.graph, to_cluster), from_node, to_node,
                                         weight='length')

    # p = nx.single_source_dijkstra(layer.graph, from_node, to_node)
    # correct_clusters = []
    # for n in p[1]:
    #     if len(correct_clusters) == 0:
    #         correct_clusters.append(layer.graph.nodes[n]['cluster'])
    #     elif correct_clusters[-1] != layer.graph.nodes[n]['cluster']:
    #         correct_clusters.append(layer.graph.nodes[n]['cluster'])

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    def dst(a, b):
        # x, y = to_d['x'], to_d['y']
        # x0, y0 = from_d['x'], from_d['y']
        # u, v = layer.graph.nodes[a]['x'], layer.graph.nodes[a]['y']
        # cls = layer.graph.nodes[a]['cluster']
        # centers = nx.barycenter(extract_cluster_subgraph(layer.graph, cls))
        # d_1 = 100000
        # for c in centers:
        #     u, v = layer.graph.nodes[c]['x'], layer.graph.nodes[c]['y']
        #     d_1 = min(d_1, ((x - u) ** 2 + (y - v) ** 2) ** 0.5 )
        return 0
    astar = False
    if paths is not None:
        print('fast get')
        path = paths[from_cluster][to_cluster]
    elif astar:
        path = nx.astar_path(
            layer.centroids_graph,
            from_center,
            to_center,
            heuristic=dst,
            weight='length')
    else:
        path = nx.single_source_dijkstra(
            layer.centroids_graph,
            from_center,
            to_center,
            weight='length')[1]
    cls = set()
    cls.add(to_cluster)
    for u in path:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)
        # if neighbour_cluster:
        #     for to in layer.cluster_to_neighboring_cluster[c]:
        #         cls.add(to)
    # print('____________')
    # a = [layer.graph.nodes[u]['cluster'] for u in path]
    # print(correct_clusters)
    # print(a)
    # print(set(a).union(set(correct_clusters)))
    g = extract_cluster_list_subgraph(layer.graph, cls)
    return nx.single_source_dijkstra(
        g,
        from_node,
        to_node,
        weight='length'
    )
