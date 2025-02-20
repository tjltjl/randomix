# randomix - Random Jax utilities

Miscellaneous things related to jax and random numbers.

For now, just the Keyer class.

(This package is an experimenting with aider, uv, and
what it takes to ship a pypi package)

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

Note that you should not pass a keyer around jit borders or vmap lambdas
or anything like that; just pass the split keys.

Open question: would it be convenient to just mirror *all* the `jax.random`
methods that take key as an argument to this class?
