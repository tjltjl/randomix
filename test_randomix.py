import pytest
import jax
import jax.random as random
from randomix_package.randomix import Keyer
from hello import hello

def test_initialization():
    # Test if the Keyer object is initialized correctly
    key = random.PRNGKey(0)
    keyer = Keyer(key)
    assert keyer.key is not None
    assert isinstance(keyer.key, jax.random.PRNGKeyArray)

def test_split_key_no_shape():
    # Test splitting a key without a shape
    key = random.PRNGKey(0)
    keyer = Keyer(key)
    new_key = keyer()
    assert new_key is not None
    assert isinstance(new_key, jax.random.PRNGKeyArray)
    assert not jax.numpy.array_equal(new_key, key)

def test_split_key_with_shape():
    # Test splitting a key with a shape
    key = random.PRNGKey(0)
    keyer = Keyer(key)
    shape = (2, 3)
    keys = keyer(shape)
    assert keys is not None
    assert isinstance(keys, tuple)
    assert len(keys) == 2  # Since we split into 2 keys
    for key in keys:
        assert isinstance(key, jax.random.PRNGKeyArray)

def test_key_uniqueness():
    # Test if generated keys are unique
    key = random.PRNGKey(0)
    keyer = Keyer(key)
    key1 = keyer()
    key2 = keyer()
    assert not jax.numpy.array_equal(key1, key2)

def test_shape_handling():
    # Test if the shape handling is correct
    key = random.PRNGKey(0)
    keyer = Keyer(key)
    shape = (3,)
    keys = keyer(shape)
    assert len(keys) == 3  # Since we split into 3 keys
    for key in keys:
        assert isinstance(key, jax.random.PRNGKeyArray)
