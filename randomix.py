import jax
import jax.random as random
from jax import dtypes

def ensure_typed_key_array(key) -> jax.Array:
    if dtypes.issubdtype(key.dtype, dtypes.prng_key):
        return key
    else:
        raise TypeError("New-style typed JAX PRNG keys required")

class Keyer:
    """
    A class to manage and split JAX PRNG keys.

    Attributes:
        key (jax.Array): The current JAX PRNG key.
    """

    def __init__(self, key):
        """
        Initializes the Keyer with a JAX PRNG key.

        Args:
            key (jax.Array): The initial JAX PRNG key.

        Raises:
            TypeError: If the provided key is not a new-style typed JAX PRNG key.
        """
        self.key = ensure_typed_key_array(key)

    def __call__(self, shape=None):
        """
        Splits the current key into sub-keys based on the provided shape.

        Args:
            shape (tuple, optional): The shape of the sub-keys to generate. If None, a single sub-key is generated.

        Returns:
            tuple: A tuple of sub-keys if a shape is provided, otherwise a single sub-key.
        """
        if shape is None:
            new_key, self.key = random.split(self.key)
            return new_key
        else:
            num_keys = jax.numpy.prod(jax.numpy.array(shape))
            keys = random.split(self.key, num_keys + 1)
            self.key = keys[-1]
            return tuple(keys[:-1])
