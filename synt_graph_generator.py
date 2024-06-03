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
    if len(sys.argv) == 1:
        number = 1
        total = 1
    else:
        number = int(sys.argv[1])
        total = int(sys.argv[2])
    f = list(fib(100))[3:]
    arr = [1 / (10 * N)]
    i = 0
    while f[i] / (10*N) < 1:
        arr.append(f[i] / (10 * N))
        i += 1
    # arr.append(1)
    #    arr = []
    #     arr.append(N)
    total_len = len(arr)
    print(arr[number - 1: total_len: total])
    for p in arr[number - 1: total_len: total]:
        G = nx.fast_gnp_random_graph(N, 1/(100*N), seed=123, directed=False)
        if not nx.is_connected(G):
            print('unconnected: {}'.format(p))
            tmp = []
            for n in nx.connected_components(G):
                for q in n:
                    tmp.append(q)
                    break
            for i in range(len(tmp) - 1):
                G.add_edge(tmp[i], tmp[i + 1])
            print('be connected: {}'.format(nx.is_connected(G)))
        N = len(G.nodes)
        M = len(G.edges)
        for e in G.edges:
            G.add_edge(e[0], e[1], length=np.random.random_sample() + 0.001)
        print(2 * M / ((N - 1) * N))
        # city_tests.test_graph_dynamic(G,
        #                               f'SYNT_random_weight_{round(betta_variation.get_density(G) * 10000) / 10000}',
        # 1000010000                              '0')
        pickle.dump(G, open('g1.pickle', 'wb'))
        break
