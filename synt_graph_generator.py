import networkx as nx
import numpy as np

import city_tests

if __name__ == '__main__':
    N = 10000
    for p in range(1, 30, 1):
        G = nx.fast_gnp_random_graph(N, p / 10000, seed=123, directed=False)
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
            G.add_edge(e[0], e[1], length=np.random.random_sample())
        # print(G.edges(data=True))
        print(2 * M / ((N - 1) * N))
        city_tests.test_graph(G, f'Synt_{p}', '0')
