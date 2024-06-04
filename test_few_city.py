import sys

import city_tests


if __name__ == '__main__':
    c = {
        'ASHA': 'R13470549',
        'KRG': 'R4676636',
        'EKB': 'R6564910',
        'BARCELONA': 'R347950',
        'PARIS': 'R71525',
        'Prague': 'R435514',
        'MSK': 'R2555133',
        'SBP': 'R337422',
        'SINGAPORE': 'R17140517',
        'BERLIN': 'R62422',
        'ROME': 'R41485',
        'LA': 'R207359',
        'DUBAI': 'R4479752',
        'RIO': 'R2697338',
        'DELHI': 'R1942586',
        'KAIR': 'R5466227'
    }

    if len(sys.argv) == 1:
        number = 1
        total = 1
    else:
        number = int(sys.argv[1])
        total = int(sys.argv[2])

    total_len = len(c)

    for city in list(c.items())[number - 1: total_len: total]:
        # H = graph_generator.get_graph(c[city])
        # print(str(len(H.nodes)) + ' ' + city)
        city_tests.test_city(*city)
        # print(*city)