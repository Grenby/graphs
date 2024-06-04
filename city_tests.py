import time

import networkx as nx
import numpy as np
from networkx import density
from tqdm import tqdm, trange

import graph_generator
from common import GraphLayer, CentroidResult, CityResult
from graph_generator import generate_layer, get_graph, get_node_for_initial_graph_v2
from pfa import find_path


def test_layer(
        H: nx.Graph,
        resolution: float,
        usual_result: list[int, list[float]],
        points: list[list[int, int]],
        layer: GraphLayer,
) -> CentroidResult:
    result = CentroidResult(
        resolution,
        len(layer.centroids_graph.nodes),
        len(layer.centroids_graph.edges),
        len(layer.centroids_graph.nodes) / len(H.nodes)
    )
    test_results = [0, []]
    start_time = time.time()
    for point_from, point_to in points:  # , desc=f'Test points {resolution}', total=len(points)):
        test_results[1].append(test_path(layer, point_from, point_to))
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
        point_to: int
) -> float:
    try:
        my_path = find_path(layer, point_from, point_to)
    except Exception as e:
        print(e)
        return -1
    return my_path[0]


def test_city(name: str, city_id: str) -> CityResult:
    print('start testing : ', city_id, ' ', name)
    return test_graph(get_graph(city_id), name, city_id)


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
    num_paths: int = 1000
    if points is None:
        points = [get_node_for_initial_graph_v2(graph) for _ in trange(num_paths, desc='generate points')]
    has_coords = 'x' in [d for u, d in graph.nodes(data=True)][0]
    usual_results = [0, []]
    start_time = time.time()
    for node_from, node_to in points:
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

    for r in tqdm(resolutions, desc='test resolutions:'):
        layer = generate_layer(graph, r, has_coordinates=has_coords)
        tmp = test_layer(graph, r, usual_results, points, layer)
        while len(tmp.errors) < num_paths // 10 * 9:
            tmp = test_layer(graph, r, usual_results, points, layer)
        result.points_results.append(tmp)
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
