import numpy as np
from distances import cosine_distances


def test_shape():
    a = np.random.rand(5, 128)
    b = np.random.rand(3, 128)
    assert cosine_distances(a, b).shape == (5, 3)


def test_self_distance_is_zero():
    a = np.random.rand(4, 128)
    d = cosine_distances(a, a)
    assert np.allclose(np.diag(d), 0)


def test_opposite_vectors():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = cosine_distances(a, -a)
    assert np.allclose(np.diag(d), 2)


def test_orthogonal():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert np.allclose(cosine_distances(a, b), 1)


if __name__ == "__main__":
    test_shape()
    test_self_distance_is_zero()
    test_opposite_vectors()
    test_orthogonal()
    print("all good")
