import time

import networkx as nx
from tqdm import tqdm, trange

from common import GraphLayer, CentroidResult, CityResult
from graph_generator import generate_layer, get_graph, get_node_for_initial_graph, get_node_for_initial_graph_v2
from pfa import find_path


def test_layer(
        H: nx.Graph,
        resolution: float,
        usual_result: list[int, list[float]],
        points: list[list[int, int]],
        add_neighbour_cluster=True
) -> CentroidResult:
    layer = generate_layer(H, resolution)

    result = CentroidResult(
        resolution,
        len(layer.centroids_graph.nodes),
        len(layer.centroids_graph.edges),
        len(layer.centroids_graph.nodes) / len(H.nodes)
    )
    test_results = [0, []]
    start_time = time.time()
    for point_from, point_to in points:  # , desc=f'Test points {resolution}', total=len(points)):
        test_results[1].append(test_path(layer, point_from, point_to, add_neighbour_cluster))
    end_time = time.time()
    test_time = end_time - start_time

    result.speed_up.append(abs(usual_result[0] / test_time))
    result.absolute_time.append(test_time)

    for i, p in enumerate(test_results[1]):
        if p == -1:
            continue
        usual_path_len = max(usual_result[1][i], 0.00001)

        result.errors.append(abs(usual_path_len - p) / usual_path_len)
        result.absolute_err.append(abs(usual_path_len - p))

    return result


def test_path(
        layer: GraphLayer,
        point_from: int,
        point_to: int,
        add_neighbour_cluster=True
) -> float:
    try:
        my_path = find_path(layer, point_from, point_to, add_neighbour_cluster)
    except:
        return -1
    return my_path[0]


def test_city(name: str, city_id: str) -> CityResult:
    print('start testing : ', city_id, ' ', name)
    H = get_graph(city_id)
    return test_graph(H, name, city_id)


def test_graph(graph: nx.Graph, name: str, city_id: str, points: list = None) -> CityResult:
    resolutions = [i for i in range(1, 10,
                                    1)]
    resolutions += [i for i in range(10, 100,
                                     5)]
    resolutions += [i for i in range(100, 500,
                                     10)]
    resolutions += [i for i in range(500, 1000,
                                     10)]
    resolutions += [i for i in range(1000, 10000, 100)]

    if points is None:
        N: int = 1000
        points = [get_node_for_initial_graph_v2(graph) for i in trange(N, desc='generate points')]

    usual_results = [0, []]
    start_time = time.time()
    for node_from, node_to in points:
        usual_path = nx.single_source_dijkstra(graph, node_from, node_to, weight='length')
        usual_results[1].append(usual_path[0])
    end_time = time.time()
    usual_time = end_time - start_time
    usual_results[0] = usual_time
    # print(usual_time / 1000000)
    result = CityResult(
        name=name,
        name_suffix='',
        city_id=city_id,
        nodes=len(graph.nodes),
        edges=len(graph.edges)
    )

    for r in tqdm(resolutions, desc='test resolutions:'):
        tmp = test_layer(graph, r, usual_results, points, False)
        while len(tmp.errors) < 900:
            print('fot graph ' + name + ' resolution' + str(r) + ' alpha' + str(tmp.alpha) + ' not found enough data')
            tmp = test_layer(graph, r, usual_results, points, True)
        result.points_results.append(tmp)
        if tmp.alpha > 0.6:
            break
    result.save()
    return result
