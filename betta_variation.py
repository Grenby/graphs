import networkx as nx
import numpy as np
from tqdm import trange, tqdm
from scipy.spatial import KDTree as kd
import graph_generator


def variation(H: nx.Graph, r: int, strategy='add') -> nx.Graph:
    if strategy == 'remove':
        return remove_variation(H, r)
    else:
        return add_variation(H, r)


def remove_variation(H: nx.Graph, precent: float, r: float) -> nx.Graph:
    pass


def get_dist(du, dv) -> float:
    d = (du['x'] - dv['x']) ** 2 + (du['y'] - dv['y']) ** 2
    d = d ** 0.5 / 360 * 2 * np.pi * 6371.01 * 1000
    return d


def add_variation(H: nx.Graph, r: int) -> nx.Graph:
    _G = H.copy()
    ids = [u for u in H.nodes()]
    points = [[d['x'], d['y']] for u, d in H.nodes(data=True)]

    tree = kd(points)
    for u, du in H.nodes(data=True):
        dists, n_ids = tree.query([du['x'], du['y']], r)
        if type(n_ids) is np.int64:
            n_ids = [n_ids]
            dists = [dists]

        for i in range(len(n_ids)):
            _id = n_ids[i]
            d = dists[i]
            if ids[_id] == u:
                continue
            _G.add_edge(u, ids[_id], length=d)
    if not nx.is_connected(_G):
        tmp = []
        for n in nx.connected_components(_G):
            for q in n:
                tmp.append(q)
                break
        for i in range(len(tmp) - 1):
            d1 = _G.nodes[tmp[i]]
            d2 = _G.nodes[tmp[i]]
            _G.add_edge(tmp[i], tmp[i + 1], length=((d1['x'] - d2['x'])**2+(d1['y'] - d2['y'])**2)**0.5)
    return _G


def get_density(H: nx.Graph) -> float:
    _e = len(H.edges)
    _v = len(H.nodes)
    return 2 * _e / (_v * (_v - 1))


if __name__ == '__main__':
    H = graph_generator.get_graph('R13470549')

    min_l = 1e10
    max_l = 0

    for u, du in tqdm(H.nodes(data=True)):
        for v, dv in H.nodes(data=True):
            if u == v:
                continue
            d = get_dist(du, dv)
            min_l = min(min_l, d)
            max_l = max(max_l, d)

    print('min: ', min_l)
    print('max: ', max_l)

    for p in trange(1, 100, 10):
        r = (max_l - min_l) * p / 100 + min_l
        G = variation(H, r)
        print(get_density(G))
    # graph_map: folium.Map = drawer.draw_on_map(G)
    # print('save')
    # graph_map.save("map_new.html")
    # print('open')
    # webbrowser.open("map_new.html")
