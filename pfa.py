import time

import networkx as nx

from common import GraphLayer
from graph_generator import extract_cluster_subgraph, extract_cluster_list_subgraph


def find_path(
        layer: GraphLayer,
        from_node: int,
        to_node: int,
        neighbour_cluster=False) -> tuple[float, list[int]]:
    from_d = layer.graph.nodes[from_node]
    to_d = layer.graph.nodes[to_node]

    from_cluster = from_d['cluster']
    to_cluster = to_d['cluster']

    if from_cluster == to_cluster:
        return nx.single_source_dijkstra(extract_cluster_subgraph(layer.graph, to_cluster), from_node, to_node,
                                         weight='length')

    from_center = layer.cluster_to_center[from_cluster]
    to_center = layer.cluster_to_center[to_cluster]

    path = nx.bidirectional_dijkstra(
        layer.centroids_graph,
        from_center,
        to_center,
        weight='length')

    cls = set()

    for u in path[1]:
        c = layer.graph.nodes[u]['cluster']
        cls.add(c)
        if neighbour_cluster:
            for to in layer.cluster_to_neighboring_cluster[c]:
                cls.add(to)
    # start_time = time.time()
    g = extract_cluster_list_subgraph(layer.graph, cls)
    # my_time = time.time() - start_time
    # print(my_time)
    return nx.bidirectional_dijkstra(
        g,
        from_node,
        to_node,
        weight='length')
