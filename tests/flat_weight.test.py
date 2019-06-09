import random
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web3 import Web3, HTTPProvider
from client import Client

if __name__ == '__main__':
    web3 = Web3(HTTPProvider('http://localhost:8545'))
    c = Client(0, None, None, web3)
    c.setup_model('perceptron')
    weights_metadata = c.weights_metadata
    factor = 1e10

    weights = {
        name: np.random.random(shape) for name, (shape, _) in weights_metadata.items()
    }
    total_size = sum(v.size for _, v in weights.items())

    flattened = c.flatten_weights(weights, factor)
    assert len(flattened) == total_size, 'flatten_weights() failed.'

    for n in flattened:
        assert int(n) == n, 'flattened should have integers'

    unflattened = c.unflatten_weights(flattened, factor)
    for key1, key2 in zip(sorted(unflattened), sorted(weights)):
        assert key1 == key2, 'invalid keys'
        unflattened_weights = unflattened[key1]
        original_weights = (weights[key2] * factor).astype(int) / factor
        assert np.array_equiv(unflattened_weights, original_weights), \
            ('unflatten_weights() failed', unflattened, weights)

    print("Tests passed!")
