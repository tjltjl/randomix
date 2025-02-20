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
            keys = random.split(self.key, jax.tree_util.tree_leaves(shape))
            self.key = keys[-1]
            return tuple(keys[:-1])
