from tqdm import trange

import betta_variation as betta_variation
import city_tests
from graph_generator import get_graph, get_node_for_initial_graph

if __name__ == '__main__':
    H = get_graph('R71525')
    points = [get_node_for_initial_graph(H) for i in trange(100, desc='generate points')]

    ps = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
          0.001, 0.0011, 0.0013, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019,
          0.002, 0.0021, 0.0023, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029,
          0.003, 0.0031, 0.0033, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039,
          0.025, 0.028,
          0.005]
    # уже проверенны:
    # 0.0004, 0.0005, 0.0006, 0.0007, 0.0008
    # 0.0009, 0.001, 0.0011, 0.0013, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0019
    r = []
    k = betta_variation.get_density(H)
    p = 0.0001
    G = H
    while k < 0.99:
        r.append(city_tests.test_graph(G, f'PARIS_{round(betta_variation.get_density(G) * 10000) / 10000}', 'R71525', points=points))
        rad = round(len(H.nodes()) * p)
        G = betta_variation.variation(H, rad)
        kk = betta_variation.variation(G)
        while k + 0.1 > kk:
            G = betta_variation.variation(H, rad)
            kk = betta_variation.variation(G)
        k = kk
        print(kk)

    # for p in ps:
    #     rad = round(len(H.nodes()) * p)
    #     G = betta_variation.variation(H, rad)
    #     k = round(betta_variation.get_density(G) * 10000) / 10000
    #
    #     r.append(city_tests.test_graph(G, f'PARIS_{k}', 'R71525', points=points))