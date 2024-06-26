import os
import pickle
import queue
import time
from multiprocessing import Pool

import networkx as nx
import numpy as np
from tqdm import tqdm, trange

import graph_generator
from common import GraphLayer, CentroidResult, CityResult
from graph_generator import generate_layer, get_graph, get_node_for_initial_graph_v2
from pfa import find_path


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


def test_layer(
        points: list[list[int, int]],
        layer: GraphLayer,
) -> tuple[float, list[float]]:
    test_paths: list[float] = []
    start_time = time.time()
    for point_from, point_to in points:
        test_paths.append(test_path(layer, point_from, point_to))
    end_time = time.time()
    test_time = end_time - start_time
    return test_time, test_paths


def test_city(name: str, city_id: str) -> CityResult:
    print('start testing : ', city_id, ' ', name)
    return test_graph_fixed_step_alpha(get_graph(city_id), name, city_id)


def get_usual_result(graph: nx.Graph, points: list[tuple[int, int]]) -> tuple[float, list[float]]:
    usual_results: list[float] = []
    start_time = time.time()
    for node_from, node_to in points:
        usual_path = nx.single_source_dijkstra(graph, node_from, node_to, weight='length')
        usual_results.append(usual_path[0])
    end_time = time.time()
    usual_time = end_time - start_time
    return usual_time, usual_results


def get_points(graph: nx.Graph, N: int) -> list[tuple[int, int]]:
    return [get_node_for_initial_graph_v2(graph) for _ in range(N)]


def generate_result(
        usual_results: tuple[float, list[float]],
        test_results: tuple[float, list[float]],
        resolution: float,
        layer: GraphLayer
) -> CentroidResult:
    test_time = test_results[0]
    result = CentroidResult(
        resolution,
        len(layer.centroids_graph.nodes),
        len(layer.centroids_graph.edges),
        len(layer.centroids_graph.nodes) / len(layer.graph.nodes)
    )
    result.speed_up.append(abs(usual_results[0] / test_time))
    result.absolute_time.append(test_time)

    for i, p in enumerate(test_results[1]):
        if p == -1:
            continue
        usual_path_len = usual_results[1][i]
        result.errors.append(abs(usual_path_len - p) / usual_path_len)
        result.absolute_err.append(abs(usual_path_len - p))
    return result


def test_graph(graph: nx.Graph, name: str, city_id: str, points: list[tuple[int, int]] = None,
               resolutions: list[float] = None) -> CityResult:
    print(name, nx.is_connected(graph))
    max_alpha = 1
    delta = 1 / 80

    if resolutions is None:
        resolutions = []
        resolutions += [i / 10 for i in range(1, 10, 1)]
        resolutions += [i for i in range(1, 10, 1)]
        resolutions += [i for i in range(10, 50, 2)]
        resolutions += [i for i in range(50, 100, 5)]
        resolutions += [i for i in range(100, 500, 10)]
        resolutions += [i for i in range(500, 1000, 50)]
        resolutions += [i for i in range(1000, 2000, 200)]

    if points is None:
        N: int = 500
        points = [get_node_for_initial_graph_v2(graph) for _ in trange(N, desc='generate points')]
    else:
        N = len(points)

    has_coords = 'x' in [d for u, d in graph.nodes(data=True)][0]

    usual_results = get_usual_result(graph, points)

    result = CityResult(
        name=name,
        name_suffix='',
        city_id=city_id,
        nodes=len(graph.nodes),
        edges=len(graph.edges)
    )

    alphas = set()

    for r in tqdm(resolutions, desc='test resolutions:', position=2):
        start = time.time()
        layer, build_communities, build_additional, build_centroid_graph = generate_layer(graph, r,
                                                                                          has_coordinates=has_coords)
        a = len(layer.communities) / len(layer.graph.nodes)
        has = False
        for curr in alphas:
            if abs(curr - a) < delta:
                has = True
                break
        if has or a > max_alpha:
            tqdm.write(f'alpha: {a} -- skip')
            if a == 1 and 1 in alphas or a > max_alpha:
                break
            else:
                continue
        alphas.add(a)
        tmp = test_layer(points, layer)
        total = time.time() - start
        # while len(tmp[1]) < N // 10 * 9:
        #     tmp = test_layer(points, layer)
        text = """
            alpha:          {:4f}
            total time:     {:.3f}
            prepare time:   {:.3f} 
                build_communities:      {:.3f}
                build_additional:       {:.3f}
                build_centroid_graph:   {:.3f}
            pfa time:       {:.3f}
        """.format(a, total, total - tmp[0], build_communities, build_additional, build_centroid_graph, tmp[0])
        tqdm.write(text)
        result.points_results.append(generate_result(usual_results, tmp, r, layer))
    result.save()
    s = [p.speed_up[0] for p in result.points_results]
    print(np.mean(result.points_results[np.argmax(s)].errors), np.std(result.points_results[0].errors))
    print(np.max(result.points_results[np.argmax(s)].errors))
    print(max(s))
    return result


def func(x: float, graph: nx.Graph):
    return len(graph_generator.resolve_communities(graph, x)) / len(graph.nodes)


def bin_search(x1, x2, target, graph: nx.Graph, delta=0.001):
    print('target', target)
    f1 = func(x1, graph)
    f2 = func(x2, graph)
    if abs(target - f1) < delta:
        return x1
    if abs(target - f2) < delta:
        return x2
    x = (x1 + x2) / 2
    f = func(x, graph)
    count = 20
    while count > 0 and abs(f - target) > delta:
        if f > target:
            x2 = x
        else:
            x1 = x
        x = (x1 + x2) / 2
        f = func(x, graph)
        count -= 1
        print('f', f)
    return x


def solver(x1, x2, target_y, graph: nx.Graph, delta=0.001):
    y1 = func(x1, graph)
    y2 = func(x2, graph)
    target_y = [q for q in target_y if y2 >= q >= y1]
    if len(target_y) == 0:
        return []
    print(y1, y2)
    # print('all',[q for q in target_y if y2 >= q >= y1])
    # print('target', target_y)

    x = (x1 + x2) / 2
    y = func(x, graph)
    ans = []
    y_l = [q for q in target_y if q <= y]
    y_r = [q for q in target_y if q > y]
    while len(y_l) == 0 or len(y_r) == 0:
        if len(y_l) == 0:
            x1 = x
        else:
            x2 = x
        x = (x1 + x2) / 2
        y = func(x, graph)
        print('y', y)
        y_l = [q for q in target_y if q <= y]
        y_r = [q for q in target_y if q > y]
    if len(y_l) > 0 and abs(y_l[-1] - y) < delta:
        ans.append(x)
        y_l = y_l[:-1]
    if len(y_r) > 0 and (abs(y_r[0] - y) < delta):
        ans.append(x)
        y_r = y_r[1:]
    if len(y_l) > 1:
        ans += solver(x1, x, y_l, graph, delta)
    elif len(y_l) == 1:
        ans.append(bin_search(x1, x, y_l[0], graph, delta))
    if len(y_r) > 1:
        ans += solver(x, x2, y_r, graph, delta)
    elif len(y_r) == 1:
        ans.append(bin_search(x, x2, y_r[0], graph, delta))
    return ans


def ff(d):
    return solver(*d)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def test_graph_fixed_alpha(graph: nx.Graph, name: str = None, city_id: str = None,
                           alpha: float | list[float] = None) -> CityResult:
    file_name = '{}_{:.4f}'.format(len(graph.nodes), nx.density(graph))
    if os.path.isfile(file_name):
        print('exist')
        return None

    delta = 0.01
    if alpha is None:
        alpha = [a for a in np.linspace(1 / len(graph.nodes), 0.2, 15)]
        alpha += [a for a in np.linspace(0.22, 0.5, 8)]
        alpha += [a for a in np.linspace(0.55, 1, 4)]
    print('start find resolutions')
    THREADS = 1
    # data = list(split(alpha, THREADS))
    # data = [[0.01, 200, d, graph, delta] for d in data]
    # with Pool(THREADS) as p:
    #     resolutions = p.map(ff, data)

    # q = []
    # for r in resolutions:
    #     q += r
    # resolutions = q
    resolutions = solver(0.01, 600, alpha, graph, delta)
    resolutions.sort()
    # return test_graph(graph, name, city_id, resolutions=resolutions)
    print(resolutions)
    # print(alpha)
    # print([len(graph_generator.resolve_communities(graph,r))/len(graph.nodes) for r in resolutions])
    file_name = '{}_{:.4f}'.format(len(graph.nodes), nx.density(graph))
    with open(file_name, 'wb') as fp:
        pickle.dump(resolutions, fp)
        fp.close()


def func1(d):
    a = d[0]
    delta = d[1]
    graph: nx.Graph = d[2]
    position = d[3]
    right_resolution = 5000
    left_resolution = 0.01

    value = len(graph_generator.resolve_communities(graph, left_resolution)) / len(graph.nodes)
    new_resolution = 0
    count = 100
    text = 'alpha: {:.4f}'.format(a)
    # with tqdm(total=count, position=position, desc=text) as progress:
    for i in range(1):
        while abs(a - value) > delta:
            # progress.update(1)
            new_resolution = (left_resolution + right_resolution) / 2
            tmp = len(graph_generator.resolve_communities(graph, new_resolution)) / len(graph.nodes)
            if tmp > a:
                right_resolution = new_resolution
            else:
                left_resolution = new_resolution
            count -= 1
            if count == 0:
                break
        # progress.update(progress.total - count)
    # print('\n','delta:',abs(a - value))
    return new_resolution


def generate_graph_fixed_alpha(graph: nx.Graph, name: str = None, city_id: str = None,
                               alpha: float | list[float] = None) -> CityResult:
    file_name = '{}_{:.4f}'.format(len(graph.nodes), nx.density(graph))
    if os.path.isfile(file_name):
        print('exist')
        return None
    delta = 0.01
    if alpha is None:
        alpha = [a for a in np.linspace(1 / len(graph.nodes), 0.2, 15)]
        alpha += [a for a in np.linspace(0.22, 0.5, 8)]
        alpha += [a for a in np.linspace(0.55, 1, 4)]
    # print(alpha)
    data = [(a, delta, graph, i % 4) for i, a in enumerate(alpha)]
    with Pool(4) as p:
        resolutions = list(tqdm(p.imap(func, data), total=len(alpha)))
        # resolutions = p.map(func, data)
    print(resolutions)
    # print(alpha)
    # print([len(graph_generator.resolve_communities(graph,r))/len(graph.nodes) for r in resolutions])
    file_name = '{}_{:.4f}'.format(len(graph.nodes), nx.density(graph))
    with open(file_name, 'wb') as fp:
        pickle.dump(resolutions, fp)
        fp.close()


def generate_graph_fixed_alpha_2(graph: nx.Graph, name: str = None, city_id: str = None,
                                 alpha: float | list[float] = None) -> CityResult:
    file_name = '{}_{:.4f}'.format(len(graph.nodes), nx.density(graph))
    if os.path.isfile(file_name):
        print('exist')
        return None
    delta = 0.01
    if alpha is None:
        alpha = [a for a in np.linspace(1 / len(graph.nodes), 0.2, 15)]
        alpha += [a for a in np.linspace(0.22, 0.5, 8)]
        alpha += [a for a in np.linspace(0.55, 1, 4)]
    # print(alpha)

    data = [(a, delta, graph, i % 4) for i, a in enumerate(alpha)]
    with Pool(4) as p:
        resolutions = list(tqdm(p.imap(func, data), total=len(alpha)))
        # resolutions = p.map(func, data)
    print(resolutions)
    # print(alpha)
    # print([len(graph_generator.resolve_communities(graph,r))/len(graph.nodes) for r in resolutions])
    file_name = '{}_{:.4f}'.format(len(graph.nodes), nx.density(graph))
    with open(file_name, 'wb') as fp:
        pickle.dump(resolutions, fp)
        fp.close()


#

def test_graph_fixed_step_alpha(graph: nx.Graph, name: str = None, city_id: str = None) -> CityResult:
    resolutions = []
    alpha = []

    right_resolution = 5000
    left_resolution = 0.01

    resolutions.append(right_resolution)
    a1 = len(graph_generator.resolve_communities(graph, left_resolution)) / len(graph.nodes)
    resolutions.append(left_resolution)
    a2 = len(graph_generator.resolve_communities(graph, right_resolution)) / len(graph.nodes)
    q = queue.Queue()
    q.put((
        (left_resolution, a1),
        (right_resolution, a2)
    ))
    alpha += [a1, a2]

    step = 0.05
    delta_step = 0.001
    min_dst = 0.00001
    print('start generate resolutions')
    while not q.empty():
        p1, p2 = q.get()
        if abs(p1[0] - p2[0]) < min_dst:
            continue
        if abs(p1[1] - p2[1]) > step:
            kk = (p1[0] + p2[0]) / 2
            c = graph_generator.resolve_communities(graph, kk)
            val = len(c) / len(graph.nodes)
            if abs(p1[1] - val) < step / 10:
                q.put(((kk, val), (p2[0], p2[1])))
            elif abs(p2[1] - val) < step / 10:
                q.put(((p1[0], p1[1]), (kk, val)))
            else:
                resolutions.append(kk)
                alpha.append(val)
                if abs(p2[1] - val) > step:
                    q.put(((kk, val), (p2[0], p2[1])))
                if abs(p1[1] - val) > step:
                    q.put(((p1[0], p1[1]), (kk, val)))
    alpha.sort()
    print(alpha)
    resolutions.sort()
    print(resolutions)
    return test_graph(graph, name, city_id, resolutions=resolutions)
