import time

from random import randrange
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
import numpy as np

def f(x):
    return x*x

if __name__ == '__main__':
    data = np.zeros((2,2,2,2))
    data[0,0,0,0] =1
    data[1, 1, 1, 1] = 2

    indx = np.array([0], dtype=np.int16)
    print(data[indx])
    data = np.array([1,2,3,4])
    indx = np.array([0,1,2], dtype=np.int16)
    print(data[indx])