# 8 gaussiennes
import random
import numpy

def eight_gauss(num_point, std_value):
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / numpy.sqrt(2), 1. / numpy.sqrt(2)),
        (1. / numpy.sqrt(2), -1. / numpy.sqrt(2)),
        (-1. / numpy.sqrt(2), 1. / numpy.sqrt(2)),
        (-1. / numpy.sqrt(2), -1. / numpy.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    dataset = []
    for i in range(num_point):
        point = numpy.random.randn(2) * std_value
        center = random.choice(centers)
        point[0] += center[0]
        point[1] += center[1]
        dataset.append(point)
    dataset = numpy.array(dataset, dtype='float32')
    dataset /= 1.414  # stdev
    return dataset

def triangle():
    dataset = [[0, 0.3],
             [0, 0.7],
             [1/numpy.sqrt(2), 0.5]]  
    dataset = numpy.array(dataset, dtype='float32')
    return dataset