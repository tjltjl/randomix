import unittest
import jax
import jax.random as random
from randomix import Keyer

class TestKeyer(unittest.TestCase):
    
    def setUp(self):
        # Initialize the Keyer object with a random key
        self.key = random.PRNGKey(0)
        self.keyer = Keyer(self.key)
    
    def test_initialization(self):
        # Test if the Keyer object is initialized correctly
        self.assertIsNotNone(self.keyer.key)
        self.assertIsInstance(self.keyer.key, jax.random.PRNGKeyArray)
    
    def test_split_key_no_shape(self):
        # Test splitting a key without a shape
        new_key = self.keyer()
        self.assertIsNotNone(new_key)
        self.assertIsInstance(new_key, jax.random.PRNGKeyArray)
        self.assertNotEqual(new_key, self.key)
    
    def test_split_key_with_shape(self):
        # Test splitting a key with a shape
        shape = (2, 3)
        keys = self.keyer(shape)
        self.assertIsNotNone(keys)
        self.assertIsInstance(keys, tuple)
        self.assertEqual(len(keys), 2)  # Since we split into 2 keys
        for key in keys:
            self.assertIsInstance(key, jax.random.PRNGKeyArray)
    
    def test_key_uniqueness(self):
        # Test if generated keys are unique
        key1 = self.keyer()
        key2 = self.keyer()
        self.assertNotEqual(key1, key2)
    
    def test_shape_handling(self):
        # Test if the shape handling is correct
        shape = (3,)
        keys = self.keyer(shape)
        self.assertEqual(len(keys), 3)  # Since we split into 3 keys
        for key in keys:
            self.assertIsInstance(key, jax.random.PRNGKeyArray)

if __name__ == '__main__':
    unittest.main()
