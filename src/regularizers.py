import numpy as np


def l1_gradient(weights, lambda_):
    return lambda_ * np.sign(weights)


def l1_loss(weights, lambda_):
    return lambda_ * np.sum(np.abs(weights))


def l2_gradient(weights, lambda_):
    return lambda_ * weights


def l2_loss(weights, lambda_):
    return 0.5 * lambda_ * np.sum(weights**2)


REGULARIZERS = {
    "l1": {"gradient": l1_gradient, "loss": l1_loss},
    "l2": {"gradient": l2_gradient, "loss": l2_loss},
    None: {"gradient": lambda w, lam: 0, "loss": lambda w, lam: 0},
}


def get_regularizer(name):
    if name is None:
        return REGULARIZERS[None]
    name = name.lower()
    if name not in REGULARIZERS:
        raise ValueError(
            f"Unknown regularizer: '{name}'. Available: ['l1', 'l2', None]"
        )
    return REGULARIZERS[name]


def compute_regularization_gradient(weights_list, reg_type, lambda_):
    if reg_type is None or lambda_ == 0:
        return [np.zeros_like(w) for w in weights_list]

    reg = get_regularizer(reg_type)
    return [reg["gradient"](w, lambda_) for w in weights_list]


def compute_regularization_loss(weights_list, reg_type, lambda_):
    if reg_type is None or lambda_ == 0:
        return 0.0

    reg = get_regularizer(reg_type)
    return sum(reg["loss"](w, lambda_) for w in weights_list)
