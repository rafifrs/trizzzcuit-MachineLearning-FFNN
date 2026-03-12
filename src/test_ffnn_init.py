import numpy as np

from ffnn import FFNN


def test_zero_initialization():
    model = FFNN([4, 8, 3], ["relu", "softmax"], "zero")

    assert len(model.weights) == 2
    assert len(model.biases) == 2

    assert model.weights[0].shape == (4, 8)
    assert model.weights[1].shape == (8, 3)
    assert model.biases[0].shape == (1, 8)
    assert model.biases[1].shape == (1, 3)

    assert np.all(model.weights[0] == 0)
    assert np.all(model.weights[1] == 0)
    assert np.all(model.biases[0] == 0)
    assert np.all(model.biases[1] == 0)


def test_uniform_initialization_default_range():
    model = FFNN([5, 4, 2], ["tanh", "sigmoid"], "uniform")

    assert np.all(model.weights[0] >= -1.0) and np.all(model.weights[0] <= 1.0)
    assert np.all(model.weights[1] >= -1.0) and np.all(model.weights[1] <= 1.0)


def test_normal_initialization_shape_and_nonzero():
    model = FFNN([6, 3, 2], ["relu", "linear"], "normal")

    assert model.weights[0].shape == (6, 3)
    assert model.weights[1].shape == (3, 2)
    assert model.biases[0].shape == (1, 3)
    assert model.biases[1].shape == (1, 2)

    assert np.any(model.weights[0] != 0)
    assert np.any(model.weights[1] != 0)


def test_invalid_config_raises():
    try:
        FFNN([4], ["relu"], "normal")
        assert False, "Expected ValueError for too few layers"
    except ValueError:
        pass

    try:
        FFNN([4, 3], ["relu", "sigmoid"], "normal")
        assert False, "Expected ValueError for activation length mismatch"
    except ValueError:
        pass

    try:
        FFNN([4, 3], ["unknown"], "normal")
        assert False, "Expected ValueError for invalid activation"
    except ValueError:
        pass

    try:
        FFNN([4, 3], ["relu"], "xavier")
        assert False, "Expected ValueError for invalid init method"
    except ValueError:
        pass


def run_all_tests():
    test_zero_initialization()
    test_uniform_initialization_default_range()
    test_normal_initialization_shape_and_nonzero()
    test_invalid_config_raises()
    print("Semua test FFNN initialization lulus.")


if __name__ == "__main__":
    run_all_tests()
