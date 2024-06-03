from collections import deque, defaultdict

import networkx as nx
from networkx.algorithms.community.louvain import _convert_multigraph, _one_level, _gen_graph
from networkx.utils import py_random_state


def modularity(G, communities, weight="weight", resolution=1):
    if not isinstance(communities, list):
        communities = list(communities)

    out_degree = in_degree = dict(G.degree(weight=weight))
    deg_sum = sum(out_degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum ** 2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    res = sum(map(community_contribution, communities))
    print(res)
    return res


@py_random_state("seed")
def louvain_communities(G, weight="weight", resolution=1, threshold=0.0000001, seed=None):
    d = louvain_partitions(G, weight, resolution, threshold, seed)
    q = deque(d, maxlen=1)
    return q.pop()


def louvain_partitions(G, weight="weight", resolution=1, threshold=0.0000001, seed=None):
    partition = [{u} for u in G.nodes()]
    if nx.is_empty(G):
        yield partition
        return
    mod = modularity(G, partition, resolution=resolution, weight=weight)
    is_directed = G.is_directed()
    if G.is_multigraph():
        graph = _convert_multigraph(G, weight, is_directed)
    else:
        graph = G.__class__()
        graph.add_nodes_from(G)
        graph.add_weighted_edges_from(G.edges(data=weight, default=1))

    m = graph.size(weight="weight")
    partition, inner_partition, improvement = _one_level(
        graph, m, partition, resolution, is_directed, seed
    )
    improvement = True
    while improvement:
        # gh-5901 protect the sets in the yielded list from further manipulation here
        yield [s.copy() for s in partition]
        new_mod = modularity(
            graph, inner_partition, resolution=resolution, weight="weight"
        )
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(
            graph, m, partition, resolution, is_directed, seed
        )


def _one_level(G, m, partition, resolution=1, seed=None):
    com2center = {}
    com2total = {}
    # считаем центройды класторов
    for i, u in enumerate(partition):
        if i not in com2center:
            com2center[i] = {'x': 0, 'y': 0}
            com2total[i] = {0}
        com2total[i] += 1
        com2center[i]['x'] += G.nodes[u]['x']
        com2center[i]['y'] += G.nodes[u]['y']
    for i in com2center:
        com2center[i]['x'] /= com2total[i]
        com2center[i]['y'] /= com2total[i]

    node2com = {u: i for i, u in enumerate(G.nodes())}  # ноды к комьюнити
    inner_partition = [{u} for u in G.nodes()]  # новая партиция
    nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}  # длины путей
    rand_nodes = list(G.nodes)  # ноды
    seed.shuffle(rand_nodes)  # шафлим

    nb_moves = 1
    improvement = False

    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:  # берем рандомную ноду
            d = G.nodes[u]  # метадата ноды
            best_mod = 0
            best_com = node2com[u]  # комюнити ноды остаться как есть
            weights2com = _neighbor_weights(nbrs[u], node2com)  # веса к соседним комьюинити
            # определяем куда лучше из соседей засунуть данную ноду
            for nbr_com, wt in weights2com.items():  # смотрим куда можно добавить ноду по соседям
                # значение новых центройд при перемещении ноды из ее текущего комьюнити в nbr_com
                m1_x = 0
                m1_y = 0
                m2_x = 0
                m2_y = 0

                new_v = -1
                if new_v > best_mod:
                    best_mod = new_v
                    best_com = nbr_com

            if best_com != node2com[u]: # если комьюнити новое
                com = G.nodes[u].get("nodes", {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def _my_level(G, m, partition, resolution=1, seed=None):
    pass


def _neighbor_weights(nbrs, node2com):
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights
