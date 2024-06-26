import random
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

plt.rc('legend', fontsize=8)  # legend fontsize
plt.rcParams.update({'font.size': 14})

N = 24  # число субъединиц
K = np.zeros((N + 1, N + 1, N + 1, N + 1))
for i in range(N + 1):
    for j in range(N + 1):
        for k in range(N + 1):
            for l in range(N + 1):
                a = i + j
                b = k + l
                # if a == 1 and b == 1:
                #   K[i,j,k,l]=1
                # if a == 2 and b == 2:
                #   K[i,j,k,l]=1
                # if (a == 4 and b == 2) or (a == 2 and b == 4):
                #   K[i,j,k,l]=1
                if a == 6 and b == 6:
                    K[i, j, k, l] = 1
                if a == 12 and b == 12:
                    K[i, j, k, l] = 1

indx = np.argwhere(K > 0.01)


def model(_C, delta_time):  # уравнения Смолуховского
    dC = np.zeros((N + 1, N + 1))
    for [i, j, k, l] in indx:
        dC[i + k, j + l] += 1 / 2 * _C[i, j] * _C[k, l] * K[i, j, k, l]
        dC[i, j] -= K[i, j, k, l] * _C[i, j] * _C[k, l]
    return _C + dC * delta_time


peaks = [0.8066787214453491, 0.7286908212636174, 0.6640536524738351, 0.6088574255963066, 0.560692479862323,
         0.517968169510449, 0.4795791583843122, 0.4447261219839642, 0.4128122473841507, 0.38338014738955173,
         0.3560715225191292, 0.33060047507555473, 0.3067352244977619, 0.28428524289616, 0.2630920407421098,
         0.24302232388731304, 0.22396294601870886, 0.20581705524736288, 0.18850106219379073, 0.1719423787555041,
         0.15607751257404356, 0.14085066964413634, 0.1262125008964556, 0.11211917907276119, 0.09853160263657085]


def gaussian(x, amplitude, mean, stddev):
    return np.sqrt(1 / (2 * np.pi)) * amplitude * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))


def smooth(peaks, num, intensity):
    if intensity == 0:
        return np.zeros(25)
    Fer_sigma = 0.011
    FerSumo_sigma = 0.024
    sigma = FerSumo_sigma * (24 - num) / 24.0 + Fer_sigma * num / 24.0
    peak_center = peaks[num]
    smoothed_peak = np.zeros(25)
    for i in range(1, 24):
        smoothed_peak[i] = gaussian(peaks[i], intensity, peak_center, sigma) * (peaks[i - 1] - peaks[i + 1]) / 2
    smoothed_peak[0] = gaussian(peaks[0], intensity, peak_center, sigma) * (peaks[0] - peaks[1])
    smoothed_peak[24] = gaussian(peaks[24], intensity, peak_center, sigma) * (peaks[23] - peaks[24])
    return smoothed_peak / sum(smoothed_peak) * intensity


def smooth_profile(p_24):
    result = np.zeros(25)
    for i in range(0, 25):
        result += smooth(peaks, i, p_24[i])
    return result


def solver(C0, TIME, STEP):
    prev = C0
    for t in np.arange(0, TIME, STEP):
        C_new = model(prev, STEP)
        prev = C_new
    return prev


def f(num):
    for i in trange(800, position=num):
        C = np.zeros((N + 1, N + 1))
        C[0, 6] = random.random()
        C[1, 5] = random.random()
        C[2, 4] = random.random()
        C[3, 3] = random.random()
        C[4, 2] = random.random()
        C[5, 1] = random.random()
        C[6, 0] = random.random()
        C /= np.sum(C)
        c0 = C

        STEP = 0.1
        T = 400  # изменяйте параметр для лучших результатов

        C = solver(C, T, STEP)

        k = N
        data = np.zeros(k + 1)
        for j in range(k + 1):
            data[j] = C[j, k - j]
            # print (c)
        data = smooth_profile(data)
        data /= np.sum(data)
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.scatter(range(0, k + 1), data, s=4,
                    label='{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(c0[0, 6], c0[1, 5], c0[2, 4],
                                                                                    c0[3, 3], c0[4, 2], c0[5, 1],
                                                                                    c0[6, 0]))
        axs.set(xlabel='i', ylabel='C')
        axs.legend(ncol=5)

        plt.savefig('plots/' + str(num) + str(i) + '_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(c0[0, 6],
                                                                                                              c0[1, 5],
                                                                                                              c0[2, 4],
                                                                                                              c0[3, 3],
                                                                                                              c0[4, 2],
                                                                                                              c0[5, 1],
                                                                                                              c0[6, 0])
                    + '.png')
        plt.clf()
        plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        thread = 0
    else:
        thread = int(sys.argv[1])
    with Pool(thread) as p:
        p.map(f, [i for i in range(thread)])
