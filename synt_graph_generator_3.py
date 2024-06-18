import pickle
import random
import sys

import networkx as nx
from tqdm import tqdm, trange

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
    print('min distance', m)
    # return Q
    if m != 0:
        print('save')
        pickle.dump(Q, open(f'rand_points{N}.pickle', 'wb'))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        number = 1
        total = 1
    else:
        number = int(sys.argv[1])
        total = int(sys.argv[2])
    dens = [0.0022981490745372685]
    while dens[-1] * 1.6 < 1:
        dens.append(dens[-1] * 1.3)
    dens.append(1)

    # G = gen(1000)
    for N in [20000, 50000]:
        with open(f'rand_points{N}.pickle', 'rb') as f:
            G = pickle.load(f)
            f.close()
        Q = G

        points_number: int = 500
        points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
                  trange(points_number, desc='generate points')]

        total_len = len(dens)
        # dens += [0.6, 0.8]
        # dens = [0.0022981490745372685]
        skip = True
        for d in tqdm(dens[number - 1: total_len: total],desc = 'test density', position=1):
            # if skip:
            #     skip = False
            #     continue
            if d > 0.05:
                break
            k = d * (N - 1)
            Q = betta_variation.add_variation(G, k)
            for u in Q.nodes:
                if u in Q[u]:
                    print('loop')
                    Q.remove_edge(u, u)
            print('dens', nx.density(Q))

            city_tests.test_graph(Q,
                                  f'PlanePoints_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                                  '0',
                                  points=points)
