import numpy as np

# ============== Activation Functions ==============


def linear(z):
    return z


def linear_derivative(z):
    return np.ones_like(z)


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    result = np.zeros_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    result[neg] = exp_z / (1.0 + exp_z)
    return result


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def softmax(z):
    if z.ndim == 1:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    else:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def softmax_derivative(z):
    s = softmax(z)
    if z.ndim == 1:
        return np.diag(s) - np.outer(s, s)
    else:
        batch_size, num_classes = s.shape
        identity = np.eye(num_classes)
        return s[:, :, None] * (identity[None, :, :] - s[:, None, :])


def softmax_derivative_vectorized(z, upstream_gradient):
    s = softmax(z)
    if z.ndim == 1:
        return s * (upstream_gradient - np.sum(s * upstream_gradient))
    else:
        sum_term = np.sum(s * upstream_gradient, axis=-1, keepdims=True)
        return s * (upstream_gradient - sum_term)


# ============== Registry ==============

ACTIVATIONS = {
    "linear": linear,
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
}

ACTIVATION_DERIVATIVES = {
    "linear": linear_derivative,
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative,
}


def get_activation(name):
    name = name.lower()
    if name not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation: '{name}'. Available: {list(ACTIVATIONS.keys())}"
        )
    return ACTIVATIONS[name]


def get_activation_derivative(name):
    name = name.lower()
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(
            f"Unknown activation: '{name}'. Available: {list(ACTIVATION_DERIVATIVES.keys())}"
        )
    return ACTIVATION_DERIVATIVES[name]
