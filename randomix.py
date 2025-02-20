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

    def get_keys(self, shape):
        return self(shape)

# Example usage:
if __name__ == "__main__":
    keyer = Keyer(random.PRNGKey(0))

    other_key = keyer()  # Split a new key, update key held inside keyer
    print("Other key:", other_key)

    more_keys = keyer((3, 2))  # Get keys with a shape
    print("More keys:", more_keys)
