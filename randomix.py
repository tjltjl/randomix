import jax
import jax.random as random
from jax import dtypes

def ensure_typed_key_array(key) -> jax.Array:
    if dtypes.issubdtype(key.dtype, dtypes.prng_key):
        return key
    else:
        raise TypeError("New-style typed JAX PRNG keys required")

class Keyer:
    def __init__(self, key):
        self.key = ensure_typed_key_array(key)

    def __call__(self, shape=None):
        if shape is None:
            new_key, self.key = random.split(self.key)
            return new_key
        else:
            num_keys = jax.tree_util.tree_structure(shape).num_leaves
            keys = random.split(self.key, num_keys)
            self.key = keys[-1]
            return tuple(keys[:-1])
