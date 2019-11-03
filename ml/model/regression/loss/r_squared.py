import numpy as np

'''
See:
https://en.wikipedia.org/wiki/Coefficient_of_determination
For calculation details.
'''
def calc_r_squared(f, y):
    y_mean = np.average(y)
    SS_tot = np.sum(np.square(y - y_mean))
    SS_res = np.sum(np.square(f - y))
    return 1.0 - (SS_res / SS_tot)
