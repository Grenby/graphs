import pickle
import random
import sys

import networkx as nx
from tqdm import tqdm, trange

import betta_variation
import city_tests
import graph_generator
import math
import random

import numpy as np
from scipy.spatial import KDTree as kd

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
    return Q
def add_density(H: nx.Graph, r) -> nx.Graph:
    _G = H.copy()
    ids = [node for node in H.nodes()]
    points = [[d['x'], d['y']] for u, d in H.nodes(data=True)]

    tree = kd(points)
    random.seed(123)
    prob = r - int(r)
    for u, du in H.nodes(data=True):

        dists, n_ids = tree.query([du['x'], du['y']], math.ceil(r))
        if type(n_ids) is np.int64:
            n_ids = [n_ids]
            dists = [dists]
        if math.ceil(r) == 1:
            total = len(n_ids)
        else:
            total = len(n_ids) - 1
            if random.random() < prob:
                total += 1
        for i in range(total):
            _id = n_ids[i]
            d = dists[i]
            if ids[_id] == u:
                continue
            _G.add_edge(u, ids[_id], length=d)
    if not nx.is_connected(_G):
        # print('fix connected')
        tmp = []
        for n in nx.connected_components(_G):
            for q in n:
                tmp.append(q)
                break
        for i in range(len(tmp) - 1):
            d1 = _G.nodes[tmp[i]]
            d2 = _G.nodes[tmp[i + 1]]
            _G.add_edge(tmp[i], tmp[i + 1], length=((d1['x'] - d2['x']) ** 2 + (d1['y'] - d2['y']) ** 2) ** 0.5)
    return _G

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

    for N in [2000,5000,10000,15000,20000,30000, 50000]:
        G = gen(N)

        points_number: int = 500
        points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
                  trange(points_number, desc='generate points')]

        total_len = len(dens)
        for d in tqdm(dens[number - 1: total_len: total],desc = 'test density', position=1):
            if d > 0.05:
                break
            k = d * (N - 1)
            Q = add_density(G, k)
            for u in Q.nodes:
                if u in Q[u]:
                    Q.remove_edge(u, u)
            print('dens', nx.density(Q))
            city_tests.test_graph(Q,
                                  f'PlanePoints_{len(G.nodes)}_{round(nx.density(Q) * 10000) / 10000}',
                                  '0',
                                  points=points)
