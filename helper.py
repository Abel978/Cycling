import numpy as np
from itertools import chain


def custom_flatten(arr):

        return np.array(list(chain.from_iterable(arr)))