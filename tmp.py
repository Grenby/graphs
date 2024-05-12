import pickle
import webbrowser
from functools import partial
from os import listdir
from os.path import isfile, join

import numpy as np
from folium import folium
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy
from tqdm import trange

import betta_variation as betta_variation
import drawer
import graph_generator
import city_tests
from common import CityResult
from file_name_generator import generate_new_name
from graph_generator import get_graph, get_node_for_initial_graph_v2, get_node_for_initial_graph
from map_drawer import draw_on_map
from tg_seldler import send_massage

if __name__ == '__main__':
    # 'BARCELONA': 'R347950',
    # 'SINGAPORE': 'R17140517',
    # 'PARIS': 'R71525',
    # 'BERLIN': 'R62422',
    # 'DUBAI': 'R4479752'
    # 'Shibuya' : 'R1759477',
    c = {'Asha': 'R13470549', 'KRG': 'R4676636', 'EKB': 'R6564910', 'MSK': 'R2555133', 'SBP': 'R337422',
         'ROME': 'R41485', 'LA': 'R207359',
         'RIO': 'R2697338', 'Prague': 'R435514'}
    H = get_graph('R13470549')
    q = graph_generator.generate_layer(H, 200)
    map: folium.Map = draw_on_map(H, communities=q.communities, cluster_to_neighboring_clusters= q.cluster_to_neighboring_cluster)
    # map = draw_on_map(H, communities=communities, m=map)
    # map = draw_on_map(P, node_colors='white', m=map)
    # map = draw_on_map(my_P, node_colors='red', m=map)

    map.save("map.html")
    webbrowser.open("map.html")
    # # print(len(H.nodes))
    # points = [get_node_for_initial_graph(H) for i in trange(1000, desc='generate points')]
    # r = [tests.test_graph(H, 'PARIS', 'R71525', points=points)]

    ps = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
          0.001, 0.0011, 0.0013, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019,
          0.002, 0.0021, 0.0023, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029,
          0.003, 0.0031, 0.0033, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039,
          0.025, 0.028,
          0.005]
    # 0.0004, 0.0005, 0.0006, 0.0007, 0.0008
    # 0.0009, 0.001, 0.0011, 0.0013, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017
    r = []
    # for p in [0.0019]:
    #     rad = round(len(H.nodes()) * p)
    #     G = betta_variation.variation(H, rad)
    #     k = round(betta_variation.get_density(G) * 10000) / 10000
    #     r.append(tests.test_graph(G, f'PARIS_{k}', 'R71525', points=points))

    # mypath = './clusters_results/2024_04_22'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for name in onlyfiles:
    #     print(name)
    #     with open(join(mypath, name), 'rb') as f:
    #         r.append(pickle.load(f))
    #         f.close()
    # mypath = './clusters_results/2024_04_23'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for name in onlyfiles:
    #     print(name)
    #     with open(join(mypath, name), 'rb') as f:
    #         r.append(pickle.load(f))
    #         f.close()
    # mypath = './clusters_results/2024_04_24'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # for name in onlyfiles:
    #     print(name)
    #     with open(join(mypath, name), 'rb') as f:
    #         r.append(pickle.load(f))
    #         f.close()
    # r = r[0:15]
    # # _, file = drawer.draw_few_city_result(r)
    #
    # x = [city.density for city in r]
    # sp = [[np.mean(k.speed_up) for k in city.points_results] for city in r]
    #
    #
    # def power_func(x, a, b):
    #     return a * x + b
    #
    #
    # def theory_func(x, c, g, N=9574, b0=1):
    #     # 17718
    #     x = x * c
    #     q = np.log(x) / np.log(N)
    #     return g / (x / b0 * (1 + q) + 1 / (2 * np.sqrt(x * N)) * (1 - q))
    #
    #
    # indx = [np.argmax(s) for s in sp]
    #
    # err = [[np.mean(k.errors) for k in city.points_results] for city in r]
    # err = [err[i][indx[i]] for i in range(len(indx))]
    # err = []
    # sp = []
    #
    # alpha = []
    #
    # for res in r:
    #     city: CityResult = res
    #     func_r = partial(theory_func, N=9574, b0=city.density)
    #     x = [k.alpha for k in city.points_results]
    #     y = [np.mean(k.speed_up) for k in city.points_results]
    #     popt = curve_fit(func_r, x, y, p0=[0.001, 4])
    #     print('сгкм аше',popt[0])
    #     func_target = partial(func_r, c=popt[0][0], g=popt[0][1])
    #
    #     a = scipy.optimize.fminbound(lambda x: -func_target(x), 0, 0.5)
    #     alpha.append(a)
    #     sp.append(func_target(a))
    #
    #     y = [np.mean(k.errors) for k in city.points_results]
    #     z = np.polyfit(x, y, 4)
    #     p = np.poly1d(z)
    #     print(p(a))
    #     err.append(p(a))
    #
    # y = sp
    # x = [city.density for city in r]
    #
    # fig, axs = plt.subplots(1, 3)
    # fig.set_figwidth(25)
    # fig.set_figheight(7)
    #
    # axs[0].errorbar(x, y, fmt='o')
    # axs[0].set(xlabel='плотность', ylabel='speed_up')
    #
    # axs[1].errorbar(x, err, fmt='o')
    # axs[1].set(xlabel='плотность', ylabel='error')
    #
    # axs[2].errorbar(x, alpha, fmt='o')
    # axs[2].set(xlabel='плотность', ylabel='argmax[speddup(alpha)]')
    #
    # popt = curve_fit(power_func, x, y, p0=[1, 1])
    # print(popt)
    # # x = np.linspace(0, 1, 100)
    # y = power_func(np.array(x), *popt[0])
    # axs[0].errorbar(x, y, linewidth=3)
    #
    # file_name = generate_new_name('плотность' + '.png', 'plots')
    # plt.savefig(file_name)
    #
    # send_massage('плотность', file_name)

    # send_massage('all', file)
