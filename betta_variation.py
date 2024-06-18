import math
import random

import networkx as nx
import numpy as np
from scipy.spatial import KDTree as kd


def get_dist(du, dv) -> float:
    d = (du['x'] - dv['x']) ** 2 + (du['y'] - dv['y']) ** 2
    d = d ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000
    return d


def add_variation(H: nx.Graph, r) -> nx.Graph:
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
