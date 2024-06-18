import random
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def solver(C0, TIME, STEP):
    C = [C0]
    times = [-STEP]
    for t in tqdm(np.arange(0, TIME, STEP)):
        C_new = model(C[-1], STEP)
        C.append(C_new)
        times.append(t)
    return np.array(C), np.array(times)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        thread = 0
    else:
        thread = sys.argv[1]
    for i in range(800):
        C = np.zeros((N + 1, N + 1))
        C[0, 6] = random.random()
        C[1, 5] = random.random()
        C[2, 4] = random.random()
        C[3, 3] = random.random()
        C[4, 2] = random.random()
        C[5, 1] = random.random()
        C[6, 0] = random.random()
        C /= np.sum(C)

        STEP = 0.001
        T = 100  # изменяйте параметр для лучших результатов

        Cs, ts = solver(C, T, STEP)
        print(np.min(Cs))
        print(np.max(Cs))

        k = N
        data = np.zeros((len(ts), k + 1))
        for i in range(len(ts)):
            c = Cs[i]
            for j in range(k + 1):
                data[i, j] = c[j, k - j]
            # print (c)

        last_index = 0
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        for i in range(0, k + 1):
            axs[0].plot(ts[last_index:], data[last_index:, i], label='F' + str(i) + 'FS' + str(k - i))
        axs[0].set(xlabel='time, ms', ylabel='C')
        axs[0].legend(ncol=5)

        # axs[0].legend() # C(t) для N-меров
        c0 = Cs[0]
        time_moment = -1
        axs[1].scatter(range(0, k + 1), data[-5, 0:], s=4,
                       label='{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}'.format(c0[0, 6], c0[1, 5], c0[2, 4],
                                                                                       c0[3, 3], c0[4, 2], c0[5, 1],
                                                                                       c0[6, 0]))
        axs[1].set(xlabel='i', ylabel='C')
        axs[1].legend(ncol=5)

        plt.savefig('plots/'+str(thread)+'_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}'.format(c0[0, 6], c0[1, 5], c0[2, 4],
                                                                                       c0[3, 3], c0[4, 2], c0[5, 1],
                                                                                       c0[6, 0]) + '.png')
        plt.cla()