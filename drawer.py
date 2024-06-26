from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, optimize

from file_name_generator import generate_new_name
import xlsxwriter

from common import CityResult, CentroidResult


def theory_func(x, c, g, N = 9574, b0 = 1):
    # 17718
    x = x * c
    q = np.log(x) / np.log(N)
    return g / (x / b0 * (1 + q) + 1 / (2 * np.sqrt(x * N)) * (1 - q))
    # return c / (x + a / np.sqrt(N * x))


def power_func(x, c, a):
    return c * (x ** (a))


def draw_few_city_result(city_result: list[CityResult]):
    fig, axs = plt.subplots(1, 2)
    fig.set_figwidth(25)
    fig.set_figheight(7)

    for i in range(len(city_result)):

        err_mean = []
        speed_up_mean = []
        alpha = []

        for r in city_result[i].points_results:
            # print(np.mean(r.errors))
            # print(np.mean(r.speed_up))
            # print('_______________________________________')
            err_mean.append(np.mean(r.errors))
            speed_up_mean.append(np.mean(r.speed_up))
            alpha.append(r.alpha)

        f = partial(theory_func,N = 9574, b0=city_result[i].density)
        popt = curve_fit(f, alpha, speed_up_mean, p0 = [0.001,4])
        print(popt)
        # x = np.linspace(0, 1, 100)
        y = f(np.array(alpha), *popt[0])
        axs[0].errorbar(alpha, y, linewidth=3)
        axs[0].errorbar(alpha, speed_up_mean, fmt='o', label=city_result[i].name)
        axs[1].errorbar(alpha, err_mean, fmt='o', label=city_result[i].name)

    axs[0].set(xlabel=r'$\alpha$', ylabel='speed_up')
    axs[1].set(xlabel=r'$\alpha$', ylabel='err_rel')

    axs[0].legend()

    axs[1].legend()
    file_name = generate_new_name('all' + '.png', 'plots1')
    return plt.savefig(file_name), file_name


def draw_city_result(city_result: CityResult):
    fig, axs = plt.subplots(1, 3)
    fig.set_figwidth(25)
    fig.set_figheight(7)

    err_mean = []
    speed_up_mean = []
    alpha = []
    for r in city_result.points_results:
        err_mean.append(np.mean(r.errors))
        speed_up_mean.append(np.mean(r.speed_up))
        alpha.append(r.alpha)

    axs[0].errorbar(alpha, speed_up_mean, fmt='o')
    axs[1].errorbar(alpha, err_mean, fmt='o')
    axs[2].errorbar(speed_up_mean, err_mean, fmt='o')

    # for i, r in enumerate(resolutions):
    #     axs[2].text(speed_up_mean[i] * 1.05, err_mean[i] * 1.05, '{0:.1f}'.format(results[r].k))

    axs[0].set(xlabel=r'$\alpha$', ylabel='speed_up')
    axs[1].set(xlabel=r'$\alpha$', ylabel='err_rel')
    axs[2].set(xlabel='speed_up', ylabel='err_rel')
    # plt.show()
    # axs[0].set_title('variable, asymmetric error')
    # axs[0].set_yscale('log')

    # popt = curve_fit(theory_func, x, speed_up_mean, p0=[1, 1])
    # print(node)
    # print(popt)
    # # x = np.linspace(0, 1, 100)
    # y = theory_func(np.array(x), *popt[0])
    # axs[0].errorbar(x, y, linewidth=3)
    file_name = generate_new_name(city_result.name + '.png', 'plots1')
    return plt.savefig(file_name), file_name
