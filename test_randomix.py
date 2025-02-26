import pytest
import jax
import jax.random as random
from randomix import Keyer

def test_initialization():
    # Test if the Keyer object is initialized correctly
    key = random.key(0)
    keyer = Keyer(key)
    assert keyer.key is not None
    assert jax.dtypes.issubdtype(keyer.key.dtype, jax.dtypes.prng_key)

def test_split_key_no_shape():
    # Test splitting a key without a shape
    key = random.key(0)
    keyer = Keyer(key)
    new_key = keyer()
    assert new_key is not None
    assert jax.dtypes.issubdtype(new_key.dtype, jax.dtypes.prng_key)
    assert not jax.numpy.array_equal(new_key, key)

def test_split_key_with_shape():
    # Test splitting a key with a shape
    key = random.key(0)
    keyer = Keyer(key)
    shape = (2, 3)
    keys = keyer(shape)
    assert keys is not None
    assert isinstance(keys, jax.Array)
    assert keys.shape == shape  # Since we split into keys with the given shape
    assert jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)

def test_key_uniqueness():
    # Test if generated keys are unique
    key = random.key(0)
    keyer = Keyer(key)
    key1 = keyer()
    key2 = keyer()
    assert not jax.numpy.array_equal(key1, key2)

def test_shape_handling():
    # Test if the shape handling is correct
    key = random.key(0)
    keyer = Keyer(key)
    shape = (3,)
    keys = keyer(shape)
    assert keys is not None
    assert isinstance(keys, jax.Array)
    assert keys.shape == shape  # Since we split into keys with the given shape
    assert jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)

def test_shape_handling_with_tuple():
    # Test if the shape handling returns an array of keys in the shape given, not a tuple
    key = random.key(0)
    keyer = Keyer(key)
    shape = (2, 3)
    keys = keyer(shape)
    assert keys is not None
    assert isinstance(keys, jax.Array)
    assert keys.shape == shape  # Since we split into keys with the given shape
    assert jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key)
