import jax
import jax.random as random

class Keyer:
    def __init__(self, key):
        self.key = key

    def __call__(self, shape=None):
        if shape is None:
            new_key, self.key = random.split(self.key)
            return new_key
        else:
            num_keys = jax.tree_util.tree_leaves(shape)
            keys = random.split(self.key, num_keys + 1)
            self.key = keys[-1]
            return tuple(keys[:-1])
