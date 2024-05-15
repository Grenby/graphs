import networkx as nx
import numpy as np

import city_tests

if __name__ == '__main__':
    N = 10000
    for p in range(4, 20, 1):
        G = nx.fast_gnp_random_graph(N, p/1000, seed=123, directed=False)
        while not nx.is_connected(G):
            G = nx.fast_gnp_random_graph(N, p/1000, seed=123, directed=False)
            print('unconnected: {}'.format(p))
        N = len(G.nodes)
        M = len(G.edges)
        for e in G.edges:
            G.add_edge(e[0],e[1], length= np.random.random_sample())
        # print(G.edges(data=True))
        print(2*M/((N-1) * N))
        city_tests.test_graph(G, f'Synt_{p}', '0')

