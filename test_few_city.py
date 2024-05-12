import city_tests
import graph_generator

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
        'RIO': 'R2697338'}
    # deli R1942586 179748
    # c = {
    #     # 'delhi': 'R1942586',
    #     'kair': 'R5466227'
    # }
    # print(c)
    for city in c:
        # H = graph_generator.get_graph(c[city])
        # print(str(len(H.nodes)) + ' ' + city)
        city_tests.test_city(city, c[city])
