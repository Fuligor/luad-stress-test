import hashlib
import random
import numpy as np

def set_seed_from_string(seed_str: str):
    """
    Generates a deterministic integer seed from a string (e.g., filename)
    and applies it to standard random and numpy random generators.
    """
    hash_obj = hashlib.md5(seed_str.encode())
    seed_int = int(hash_obj.hexdigest(), 16) % (2**32)
    random.seed(seed_int)
    np.random.seed(seed_int)