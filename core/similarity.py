import numpy as np

def cosine_distances(descriptors_a, descriptors_b):
    """
    descriptors_a : shape-(M, D) array
    descriptors_b : shape-(N, D) array

    Returns shape-(M, N) array of cosine distances between every pair.
    0 means identical direction, 2 means opposite.
    """
    a = descriptors_a / np.linalg.norm(descriptors_a, axis=1, keepdims=True)
    b = descriptors_b / np.linalg.norm(descriptors_b, axis=1, keepdims=True)
    return 1 - a @ b.T
