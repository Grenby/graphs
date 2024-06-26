import pickle
import sys

import networkx as nx
import numpy as np
from networkx import density

import betta_variation
import city_tests


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


if __name__ == '__main__':
    N = 2000
    f = list(fib(100))[3:]
    arr = [1 / (10 * N)]
    i = 0
    while f[i] / (10*N) < 1:
        arr.append(f[i] / (10 * N))
        i += 1
    # arr.append(1)
    #    arr = []
    #     arr.append(N)
    G = nx.fast_gnp_random_graph(N, 0.01, seed=123, directed=False)
    if not nx.is_connected(G):
        tmp = []
        for n in nx.connected_components(G):
            for q in n:
                tmp.append(q)
                break
        for i in range(len(tmp) - 1):
            G.add_edge(tmp[i], tmp[i + 1])
    N = len(G.nodes)
    M = len(G.edges)
    for e in G.edges:
        G.add_edge(e[0], e[1], length=np.random.random_sample() + 0.001)
    print(2 * M / ((N - 1) * N))
    city_tests.test_graph(G,
                          f'SYNT_random_weight_{round(nx.density(G) * 10000) / 10000}', '0')
