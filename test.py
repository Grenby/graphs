if __name__ == '__main__':
    print(tuple(dict(sorted({1: [1, 2], 2: [3, 4], 0: [5, 6]}.items())).values()))
