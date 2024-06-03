import networkx as nx
from folium import DivIcon
from matplotlib import pyplot as plt
import folium
import random


def get_color_list(color_len: int):
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i / color_len) for i in range(color_len)]
    random.shuffle(colors)
    hex_colors = ['#' + ''.join([f'{int(c * 255):02x}' for c in color[:3]]) \
                  for color in colors]
    return hex_colors


def get_color_list_smart(_G: nx.Graph,
                         communities: tuple | list,
                         cluster_to_neighboring_clusters: dict[int, list[int]]) -> list[str]:
    color_len = max([len(cluster_to_neighboring_clusters[i]) for i in cluster_to_neighboring_clusters])
    colors = [i for i in range(color_len)]

    cmap = plt.get_cmap()
    cmap_colors = [cmap(i / color_len) for i in range(color_len)]
    # print(color_len)
    cmap_colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.627451, 0.1254902, 0.9411765),
        (0.0, 0.0, 1.0),
        (0.54509807, 0.27058825, 0.07450981),
        (0.93333334, 0.50980395, 0.93333334)
    ]
    hex_colors = ['#' + ''.join([f'{int(c * 255):02x}' for c in color[:3]]) \
                  for color in cmap_colors]

    cluster_to_color = {}
    result = []
    for i, c in enumerate(communities):
        free_colors = set(colors)
        for j in cluster_to_neighboring_clusters[i]:
            if j in cluster_to_color and cluster_to_color[j] in free_colors:
                free_colors.remove(cluster_to_color[j])

        color = list(free_colors)[0]

        cluster_to_color[i] = color
        result.append(hex_colors[color])

    return result


def draw_on_map(_G: nx.Graph,
                communities: tuple | list = None,
                m: folium.Map = None,
                node_colors: list | str = None,
                edge_colors: str = 'black',
                r=0,
                cluster_to_neighboring_clusters: dict[int, list[int]] = None) -> folium.Map:
    if communities is None:
        communities = [_G.nodes()]
    if node_colors is None:
        if cluster_to_neighboring_clusters is None:
            node_colors = get_color_list(len(communities))
        else:
            node_colors = get_color_list_smart(_G, communities, cluster_to_neighboring_clusters)
    if m is None:
        u_x, u_y = None, None
        for u, d in _G.nodes(data=True):
            u_x, u_y = d['x'], d['y']
            break
        m = folium.Map(location=[u_y, u_x], zoom_start=10, tiles="https://api.mapbox.com/v4/mapbox.streets/{z}/{x}/{y}.png?access_token=mytoken",
    attr="Mapbox attribution")  # Координаты города
    if not (edge_colors is None):
        for u, v, data in _G.edges(data=True):
            u_x, u_y = _G.nodes[u]['x'], _G.nodes[u]['y']
            v_x, v_y = _G.nodes[v]['x'], _G.nodes[v]['y']
            folium.PolyLine([(u_y, u_x), (v_y, v_x)], color=edge_colors, weight=1).add_to(m)

    for i, community in enumerate(communities):
        for node in community:
            if node not in _G.nodes():
                continue
            node_data = _G.nodes[node]
            popup_text = f"Кластер: {i}, \n" + f"номер: {node}"
            folium.Circle(
                location=(node_data['y'], node_data['x']),
                radius=60,
                weight=1,
                color=node_colors[i] if isinstance(node_colors, list) else node_colors,
                fill=False,
                fill_color=node_colors[i] if isinstance(node_colors, list) else node_colors,
                fill_opacity=1,
                popup=popup_text
            ).add_to(m)
            folium.map.Marker(
                [node_data['y'] , node_data['x']],
                icon=DivIcon(
                    icon_size=(15, 3.6),
                    icon_anchor=(0, 0),
                    html='<div style="font-size: 10pt">%s</div>' % str(node_data['cluster']),
                )
            ).add_to(m)
    return m
# radius=10,
#                 color="black",
#                 weight=1,
#                 fill_opacity=0.6,
#                 opacity=1,
#                 fill_color="green",
#                 fill=False,  # gets overridden by fill_color
#                 popup="{} meters".format(10),
#                 tooltip="I am in meters",
