import pickle
from os import listdir
from os.path import isfile, join
import pandas as pd
from common import CityResult, CentroidResult

if __name__ == '__main__':
    mypath = './clusters_results/paris_variation'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    r = []
    for name in onlyfiles:
        with open(join(mypath, name), 'rb') as f:
            r.append(pickle.load(f))
            f.close()






    # results['name'] = {'a': 1, 'b': 2}
    #
    # df = pd.DataFrame.from_dict(results, orient='index', columns=['a', 'b'])
    # print(df)
