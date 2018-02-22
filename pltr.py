import numpy as np
import matplotlib.pyplot as plt


def file_to_es(name):
    ergs=[0]
    with open(name) as f:
        for x in f:
            ergs.append(ergs[-1]+int(x))
    return ergs
