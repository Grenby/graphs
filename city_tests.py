import sys
import time
import webbrowser

import networkx as nx
import numpy as np
from folium import folium
from matplotlib import pyplot as plt
from networkx import density
from tqdm import tqdm, trange

import graph_generator
from common import GraphLayer, CentroidResult, CityResult
from graph_generator import generate_layer, get_graph, get_node_for_initial_graph, get_node_for_initial_graph_v2
from map_drawer import draw_on_map
from pfa import find_path

import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def test_layer(
        H: nx.Graph,
        resolution: float,
        usual_result: list[int, list[float]],
        points: list[list[int, int]],
        layer: GraphLayer,
        add_neighbour_cluster=True,
) -> CentroidResult:
    result = CentroidResult(
        resolution,
        len(layer.centroids_graph.nodes),
        len(layer.centroids_graph.edges),
        len(layer.centroids_graph.nodes) / len(H.nodes)
    )
    p = None
    print('start:', len(layer.centroids_graph.nodes))
    p = {}
    # for n, (dst, path) in tqdm(nx.all_pairs_dijkstra(layer.graph, weight='length'), total=len(layer.graph.nodes)):
    #     cls = layer.graph.nodes[n]['cluster']
    #     if cls in p:
    #         continue
    #     p[cls] = {}
    #     for to in path:
    #         to_cls = layer.graph.nodes[to]['cluster']
    #         if to_cls in p[cls]:
    #             continue
    #         p[cls][to_cls] = set([layer.cluster_to_center[layer.graph.nodes[u]['cluster']] for u in path[to]])
    print(get_size(p)/1024/1024)
    p = {du['cluster']: {dv['cluster']: set(layer.cluster_to_center[layer.graph.nodes[pp]['cluster']] for pp in nx.dijkstra_path(layer.graph, u, v, weight='length')) for v, dv in layer.centroids_graph.nodes(data=True) if v != u} for u, du in
         tqdm(layer.centroids_graph.nodes(data=True))}
    # N = len(p) **2
    # total = sum([sum([len(p[u][v]) for v in p[u]]) for u in p])
    # print(total/N)
    test_results = [0, []]
    start_time = time.time()
    for point_from, point_to in tqdm(points):  # , desc=f'Test points {resolution}', total=len(points)):
        test_results[1].append(test_path(layer, point_from, point_to, add_neighbour_cluster, p))
    end_time = time.time()
    test_time = end_time - start_time

    result.speed_up.append(abs(usual_result[0] / test_time))
    result.absolute_time.append(test_time)

    for i, p in enumerate(test_results[1]):
        if p == -1:
            continue
        usual_path_len = usual_result[1][i]
        result.errors.append(abs(usual_path_len - p) / usual_path_len)
        result.absolute_err.append(abs(usual_path_len - p))

    return result


def test_path(
        layer: GraphLayer,
        point_from: int,
        point_to: int,
        add_neighbour_cluster=True,
        p=None
) -> float:
    my_path = find_path(layer, point_from, point_to, add_neighbour_cluster, p)
    return my_path[0]


def test_city(name: str, city_id: str) -> CityResult:
    print('start testing : ', city_id, ' ', name)
    H = get_graph(city_id)
    return test_graph(H, name, city_id)


def test_graph(graph: nx.Graph, name: str, city_id: str, points: list = None) -> CityResult:
    resolutions = []
    resolutions += [i for i in range(1, 10,
                                     1)]
    resolutions += [i for i in range(10, 100,
                                     5)]
    resolutions += [i for i in range(100, 500, 10)]

    resolutions += [i for i in range(500, 1000,
                                     10)]
    resolutions += [i for i in range(1000, 10000, 100)]

    if points is None:
        N: int = 1000
        points = [get_node_for_initial_graph_v2(graph) for _ in trange(N, desc='generate points')]
    has_coords = 'x' in [d for u, d in graph.nodes(data=True)][0]
    usual_results = [0, []]
    start_time = time.time()
    for node_from, node_to in tqdm(points, desc='usual paths'):
        usual_path = nx.single_source_dijkstra(graph, node_from, node_to, weight='length')
        usual_results[1].append(usual_path[0])
    end_time = time.time()
    usual_time = end_time - start_time
    usual_results[0] = usual_time

    result = CityResult(
        name=name,
        name_suffix='',
        city_id=city_id,
        nodes=len(graph.nodes),
        edges=len(graph.edges)
    )
    # resolutions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # resolutions = [4]

    for r in tqdm(resolutions, desc='test resolutions:'):
        layer = generate_layer(graph, r, has_coordinates=has_coords)
        tmp = test_layer(graph, r, usual_results, points, layer, add_neighbour_cluster=True)
        result.points_results.append(tmp)
        if tmp.alpha > 0.5:
            break
    result.save()

    return result


def test_graph_dynamic(graph: nx.Graph, name: str, city_id: str, points: list = None) -> CityResult:
    result = CityResult(
        name=name,
        name_suffix='',
        city_id=city_id,
        nodes=len(graph.nodes),
        edges=len(graph.edges)
    )

    right_resolution = 1000
    left_resolution = 0.01

    c_l = graph_generator.resolve_communities(graph, left_resolution)
    a_l = len(c_l) / len(graph.nodes)

    a = a_l
    a_c = c_l
    new_resolution = 0
    count = 40
    print('start')
    if density(graph) == 1:
        a = 1 / len(graph.nodes)
        count = 1
    target = 0.05
    while abs(a - target) > 0.001:
        new_resolution = (left_resolution + right_resolution) / 2
        a_c = graph_generator.resolve_communities(graph, new_resolution)
        a = len(a_c) / len(graph.nodes)
        if a > target:
            right_resolution = new_resolution
        else:
            left_resolution = new_resolution
        print('alpha:', a, ' resolution:', new_resolution)
        count -= 1
        if count == 0:
            break
    print('alpha:', a, ' resolution:', new_resolution)

    layer = graph_generator.generate_layer(graph, new_resolution)
    # for cls in layer.cluster_to_bridge_points:
    #     print('clusters:', cls, ': ', len(layer.cluster_to_bridge_points[cls]) / len(layer.communities[cls]),
    #           'nodes:',len(layer.communities[cls]), 'neigh:', len(layer.cluster_to_bridge_points[cls]))
    # x = [len(layer.cluster_to_bridge_points[cls]) / len(layer.communities[cls]) for cls in
    #      layer.cluster_to_bridge_points]
    # plt.new_figure_manager()
    # plt.hist(x, label='Fixed: {:.4f}'.format(nx.density(graph)))
    # plt.legend()
    # plt.xlabel('отношение количества граничных точек кластера ко всем точкам кластера')
    # plt.ylabel('количество кластеров')
    # plt.savefig(f'{nx.density(graph)}.png')
    # map: folium.Map = draw_on_map(graph, communities=layer.communities)
    # map = draw_on_map(H, communities=communities, m=map)
    # map = draw_on_map(P, node_colors='white', m=map)
    # map = draw_on_map(my_P, node_colors='red', m=map)
    # map.save("map.html")
    # webbrowser.open("map.html")

    for i in trange(1):
        N: int = 500
        points = [get_node_for_initial_graph_v2(graph) for i in range(N)]

        usual_results: list[int, list[float]] = [0, []]
        start_time = time.time()
        for node_from, node_to in tqdm(points):
            usual_path = nx.single_source_dijkstra(graph, node_from, node_to, weight='length')
            usual_results[1].append(usual_path[0])
        end_time = time.time()
        print(np.mean(usual_results[1]))
        usual_time = end_time - start_time
        usual_results[0] = usual_time

        if abs(a - 1 / len(graph.nodes)) < 0.0001:
            print('resolution:', 10)
            result.points_results.append(test_layer(graph, 10, usual_results, points, layer))
        else:
            print('resolution:', new_resolution)
            result.points_results.append(test_layer(graph, new_resolution, usual_results, points, layer))
    result.save()
    print(np.mean(result.points_results[0].errors), np.std(result.points_results[0].errors))
    print(np.max(result.points_results[0].errors))
    print(result.points_results[0].speed_up)
    return result
