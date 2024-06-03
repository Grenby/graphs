import pickle
import random
import sys

import networkx as nx
import numpy as np
from tqdm import tqdm

import city_tests
import graph_generator


def var(G, _p: dict[int, dict[int, float]], e, pairs):
    _Q = G.copy()
    nodes = random.sample(pairs, min(e, len(pairs)))
    for _from, _to in nodes:
        _Q.add_edge(_from, _to, length=_p[_from][_to])
    return _Q


if __name__ == '__main__':
    with open('g1.pickle', 'rb') as f:
        G = pickle.load(f)
        f.close()
    Q = G

    #
    _p: dict[int, dict[int, float]] = dict(nx.all_pairs_dijkstra_path_length(Q, weight='length'))
    u = []
    N = len(G.nodes)
    for i in range(N):
        for j in range(i + 1, N):
            u.append((i, j))

    f = [0.0022981490745372685]
    while f[-1] * 1.3< 0.5:
        f.append(f[-1] * 1.3)
    f.append(1)

    if len(sys.argv) == 1:
        number = 1
        total = 1
    else:
        number = int(sys.argv[1])
        total = int(sys.argv[2])

    total_len = len(f)
    print(f[number - 1: total_len: total])
    f = [0.6,0.8]
    for d in f[number - 1: total_len: total]:
        Q = var(G, _p, round(d / 2 * N * (N - 1)), u)
        print(nx.density(Q))

        city_tests.test_graph_dynamic(Q,
                                      f'G1_random_weight_and_paths{round(nx.density(G) * 10000) / 10000}',
                                      '0')