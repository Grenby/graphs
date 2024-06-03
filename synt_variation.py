import pickle
import sys
from os import listdir
from os.path import isfile, join

import networkx as nx
from tqdm import trange

import betta_variation as betta_variation
import city_tests
from graph_generator import get_graph, get_node_for_initial_graph

if __name__ == '__main__':
    # H = get_graph('R6564910')
    # points = [get_node_for_initial_graph(H) for i in trange(30, desc='generate points')]
    #
    # # ps = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
    #       0.001, 0.0011, 0.0013, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019,
    #       0.002, 0.0021, 0.0023, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029,
    #       0.003, 0.0031, 0.0033, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039,
    #       0.025, 0.028,
    #       0.005]
    # # уже проверенны:
    # # 0.0004, 0.0005, 0.0006, 0.0007, 0.0008
    # # 0.0009, 0.001, 0.0011, 0.0013, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0019
    # # r = []
    # k = betta_variation.get_density(H)
    # p = 0.04
    # G = H
    # while k < 0.99:
    #     rad = round(len(H.nodes()) * p)
    #     kk = betta_variation.get_density(G)
    #     while k + 0.05 > kk:
    #         p += 0.04
    #         rad = round(len(H.nodes()) * p)
    #         G = betta_variation.variation(H, rad)
    #         kk = betta_variation.get_density(G)
    #     k = kk
    #     print(kk)
    #     pickle.dump(G, open(f'EKB{round(betta_variation.get_density(G) * 10000) / 10000}.pickle', 'wb'))
    #
    # G = betta_variation.variation(H, len(H.nodes))
    # pickle.dump(G, open(f'EKB{round(betta_variation.get_density(G) * 10000) / 10000}.pickle', 'wb'))
    # r.append(city_tests.test_graph(G, f'EKB_{round(betta_variation.get_density(G) * 10000) / 10000}', 'R6564910',
    #                                points=points))
    # my_path = './synt/'
    # onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    # for n in onlyfiles:
    #     print('\'' + n + '\',')
    #
    # graphs: list[nx.Graph] = []
    # for name in onlyfiles:
    #     with open(join(my_path, name), 'rb') as f:
    #         graphs.pickle.load(f)
    #         f.close()

    names = [
        'SYNT0.7006.pickle',
        'SYNT0.2504.pickle',
        'SYNT0.0505.pickle',
        'SYNT0.9007.pickle',
        'SYNT0.002.pickle',
        'SYNT0.2004.pickle',
        'SYNT0.4003.pickle',
        'SYNT0.5501.pickle',
        'SYNT0.5001.pickle',
        'SYNT0.6502.pickle',
        'SYNT0.7502.pickle',
        'SYNT0.8004.pickle',
        'SYNT0.4501.pickle',
        'SYNT0.8504.pickle',
        'SYNT0.1006.pickle',
        'SYNT0.9508.pickle',
        'SYNT0.1502.pickle',
        'SYNT0.2999.pickle',
        'SYNT0.6002.pickle',
        'SYNT0.35.pickle'
    ]
    for name in names:
        G = None
        with open(join('ekb_var/', name), 'rb') as f:
            G = pickle.load(f)
            f.close()
        for e in G.edges:
            G.add_edge(e[0], e[1], length=1)

        city_tests.test_graph(G, f'SYNT_var_{round(betta_variation.get_density(G) * 10000) / 10000}', '0', points=points)
