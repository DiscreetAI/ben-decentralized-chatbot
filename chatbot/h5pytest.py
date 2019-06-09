import h5py
import numpy as np

f = h5py.File('my_model_weights.h5', 'r')
keys = list(f.keys())
layers = [f.get(key) for key in list(f.keys())]
# print(layers)
groups = [layer.items() for layer in layers]
group1 = groups[0]
print(group1.items())
# [print(np.array(layer).shape) for layer in layers]