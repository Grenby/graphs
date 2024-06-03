import pickle
import random
import sys

import networkx as nx
import numpy as np
from tqdm import tqdm

import betta_variation
import city_tests
import graph_generator


def var(G, _p: dict[int, dict[int, float]], e, pairs):
    _Q = G.copy()
    nodes = random.sample(pairs, min(e, len(pairs)))
    for _from, _to in nodes:
        _Q.add_edge(_from, _to, length=_p[_from][_to])
    return _Q


def gen(N):
    W, H = 1000, 1000
    Q = nx.Graph()
    for i in range(N):
        x = random.random() * W
        y = random.random() * H
        Q.add_node(i, x=x, y=y)

    m = 100000
    for u, d in Q.nodes(data=True):
        for j, t in Q.nodes(data=True):
            if u == j:
                continue
            dd = (d['x'] - t['x']) ** 2 + (d['y'] - t['y']) ** 2
            m = min(m, dd)
    if m != 0:
        print('save')
        pickle.dump(Q, open('rand_points.pickle', 'wb'))


if __name__ == '__main__':
    dens = [0.0022981490745372685]
    while dens[-1] * 1.6 < 1:
        dens.append(dens[-1] * 1.3)
    dens.append(1)

    # gen(2000)
    with open('rand_points.pickle', 'rb') as f:
        G = pickle.load(f)
        f.close()
    Q = G


    #
    # #
    # # #
    # # _p: dict[int, dict[int, float]] = dict(nx.all_pairs_dijkstra_path_length(Q, weight='length'))
    # # u = []
    # # N = len(G.nodes)
    # # for i in range(N):
    # #     for j in range(i + 1, N):
    # #         u.append((i, j))
    # #
    # # f = [0.0022981490745372685]
    # # while f[-1] * 1.3< 0.5:
    # #     f.append(f[-1] * 1.3)
    # # f.append(1)
    # #
    if len(sys.argv) == 1:
        number = 1
        total = 1
    else:
        number = int(sys.argv[1])
        total = int(sys.argv[2])
    #
    total_len = len(dens)
    print(dens[number - 1: total_len: total])
    dens += [0.6, 0.8]
    left = 1
    right = 2000
    prev = 0
    dens = [0.005]
    for d in dens[number - 1: total_len: total]:
        left = 1
        right = 2000

        Q = betta_variation.add_variation(G, round(left))
        count = 10
        while abs(nx.density(Q) - d) > 0.00001:
            Q = betta_variation.add_variation(G, round((right + left) / 2))
            if nx.density(Q) > d:
                right = (right + left) / 2 + 1
            else:
                left = max((right + left) / 2 - 1, 1)
            count -= 1
            if count == 0:
                break
        if abs(prev - nx.density(Q)) < 0.0001:
            continue
        print('dens:', nx.density(Q))
        prev = nx.density(Q)
        Q = graph_generator.get_graph("R6564910")
        print(len(Q.nodes))
        city_tests.test_graph_dynamic(Q,
                                      f'PlanePoints_random_weight_and_paths{round(nx.density(G) * 10000) / 10000}',
                                      '0')
        break