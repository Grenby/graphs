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


def get_b():
    b = np.zeros(25)
    b[0] = 0.015
    b[1] = 0.025
    b[2] = 0.038
    b[3] = 0.05
    b[4] = 0.061
    b[5] = 0.068
    b[6] = 0.068
    b[7] = 0.065
    b[8] = 0.061
    b[9] = 0.059
    b[10] = 0.059
    b[11] = 0.061
    b[12] = 0.055
    b[13] = 0.052
    b[14] = 0.047
    b[15] = 0.043
    b[16] = 0.036
    b[17] = 0.03
    b[18] = 0.025
    b[19] = 0.02
    b[20] = 0.017
    b[21] = 0.015
    b[22] = 0.011
    b[23] = 0.009
    b[24] = 0.007

    b /= np.sum(b)
    return b


def solver(C0, TIME, STEP):
    prev = C0
    # times = [-STEP]
    for t in np.arange(0, TIME, STEP):
        C_new = model(prev, STEP)
        prev = C_new
    return prev

def count(state):
    C = np.zeros((N + 1, N + 1))
    C[0, 6] = state[0]
    C[1, 5] = state[1]
    C[2, 4] = state[2]
    C[3, 3] = state[3]
    C[4, 2] = state[4]
    C[5, 1] = state[5]
    C[6, 0] = state[6]

    C /= np.sum(C)

    c = solver(C, T, STEP)

    data = np.zeros(N + 1)
    for j in range(N + 1):
        data[j] = c[j, N - j]
    return np.sum(((data - get_b()) ** 2) / b), state

if __name__ == '__main__':
    if len(sys.argv) != 2:
        thread = 0
    else:
        thread = sys.argv[1]

    STEP = 0.001
    T = 100
    b = get_b()
    min_delta = 1000000
    min_data = None
    batch_size = 16
    best = 5
    multy = 2
    res = {}
    with Pool(5) as p:
        for i in trange(30):
            C = np.zeros((N + 1, N + 1))
            if len(res) == 0:
                batch = np.random.random((batch_size, 7))
            else:
                batch = np.random.random((batch_size, 7))
                for num, k in enumerate(res):
                    for j in range(multy):
                        for l in range(7):
                            batch[multy * num + j, l] = min(max(res[k][l] + random.random() / 20 - 1/40, 0), 1)
            tmp = p.map(count, batch)
            for t in tmp:
                res[t[0]] = t[1]
            #
            # for state in batch:
            #     C[0, 6] = state[0]
            #     C[1, 5] = state[1]
            #     C[2, 4] = state[2]
            #     C[3, 3] = state[3]
            #     C[4, 2] = state[4]
            #     C[5, 1] = state[5]
            #     C[6, 0] = state[6]
            #
            #     C /= np.sum(C)
            #
            #     c = solver(C, T, STEP)
            #
            #     data = np.zeros(N + 1)
            #     for j in range(N + 1):
            #         data[j] = c[j, N - j]
            #     res[np.sum(((data - b) ** 2) / b)] = state
            q = list(res.items())
            q.sort(key=lambda x: x[0])
            res = dict(q[0:best])

            q = list(res.items())
            q.sort(key=lambda x: x[0])
            q = q[0:1]
            print('delta:', q[0][0])
            print('c:', q[0][1])