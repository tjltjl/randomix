# randomix - Random Jax utilities

Miscellaneous things related to jax and random numbers.

Initially, just the Keyer class.

## Keyer

The `randomix.Keyer` class handles jax.random.PRNGKey state
storage within a single jit context.

Similar to e.g. [treex.KeySeq](https://cgarciae.github.io/treex/api/KeySeq/)
but slightly more convenient API. (I've seen this basic concept
in other places as well).

Usage:

        keyer = randomix.Keyer(key)

        other_key = keyer()  # Split a new key, update key held inside keyer

        more_keys = keyer((3, 2))  # Get keys with a shape

Open question: would it be convenient to just mirror *all* the `jax.random`
methods that take key as an argument to this class?

