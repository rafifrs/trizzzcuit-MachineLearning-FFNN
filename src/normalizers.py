import numpy as np


def rmsnorm_forward(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_sq_mean = np.mean(x**2, axis=-1, keepdims=True)
    rms = np.sqrt(x_sq_mean + eps)
    x_norm = x / rms
    return x_norm * gamma


def rmsnorm_backward(
    dout: np.ndarray, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    x_sq_mean = np.mean(x**2, axis=-1, keepdims=True)
    rms = np.sqrt(x_sq_mean + eps)
    x_norm = x / rms

    # Gradient w.r.t gamma
    dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)

    # Gradient w.r.t x
    dy = dout * gamma
    mean_dy_x = np.mean(dy * x, axis=-1, keepdims=True)
    dx = (dy / rms) - (x * mean_dy_x) / (rms**3)

    return dx, dgamma
