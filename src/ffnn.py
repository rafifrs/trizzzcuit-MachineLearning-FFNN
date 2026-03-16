from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from activations import (
    get_activation,
    get_activation_derivative,
    softmax_derivative_vectorized,
)
from losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
from regularizers import compute_regularization_gradient, compute_regularization_loss
from normalizers import rmsnorm_backward, rmsnorm_forward


class FFNN:
    VALID_ACTIVATIONS = {"linear", "relu", "leaky_relu", "sigmoid", "tanh", "swish", "softmax"}
    VALID_INIT_METHODS = {"zero", "uniform", "normal", "xavier", "he"}

    def __init__(
        self,
        layer_numbers_list: List[int],
        # banyaknya layer, termasuk input layer. Dibuatnya biar awal adalah input, akhir adalah output, tengah ada hidden.
        activation_function_list: List[str],
        # banyaknya = layer_number_list -1, merupakan fungsi aktivasi dari setiap layer
        weight_initial: str = "normal",
        # tata cara inisialisasi bobot.
        # - zero: semua bobot jadi nol
        # - uniform: bobot jadi distribusi uniform
        # - normal: bobot jadi distribusi normal
        use_rmsnorm: bool = False,
        rmsnorm_eps: float = 1e-8,
    ):

        if len(layer_numbers_list) < 2:
            raise ValueError("Tidak valid, minimal ada 2 layer (input dan output).")
        if len(activation_function_list) != len(layer_numbers_list) - 1:
            raise ValueError(
                "Jumlah fungsi aktivasi harus sama dengan jumlah layer - 1."
            )
        for act in activation_function_list:
            if act not in self.VALID_ACTIVATIONS:
                raise ValueError(
                    f"{act} bukan fungsi aktivasi yang valid. Pilih dari {self.VALID_ACTIVATIONS}."
                )
        if weight_initial not in self.VALID_INIT_METHODS:
            raise ValueError(
                f"{weight_initial} bukan metode inisialisasi yang valid. Pilih dari {self.VALID_INIT_METHODS}."
            )

        self.layer_sizes = layer_numbers_list
        self.activations = activation_function_list
        self.num_layers = len(layer_numbers_list)
        self.use_rmsnorm = use_rmsnorm
        self.rmsnorm_eps = rmsnorm_eps

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.grad_weights: List[np.ndarray] = []
        self.grad_biases: List[np.ndarray] = []
        self.rms_gamma: List[np.ndarray] = []
        self.grad_rms_gamma: List[np.ndarray] = []

        self._init_weights(weight_initial)

    def _init_weights(self, method: str) -> None:
        if method == "zero":
            self._init_zero()
        elif method == "uniform":
            self._init_uniform(
                lower=-1.0,
                upper=1.0,
                seed=None,
            )
        elif method == "normal":
            self._init_normal(
                mean=0.0,
                variance=1.0,
                seed=None,
            )
        elif method == "xavier":
            self._init_xavier(seed=None)
        elif method == "he":
            self._init_he(seed=None)

    def _init_zero(self) -> None:
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            self.weights.append(np.zeros((fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))
            if self.use_rmsnorm:
                self.rms_gamma.append(np.ones((1, fan_out)))
                self.grad_rms_gamma.append(np.zeros((1, fan_out)))

    def _init_uniform(self, lower: float, upper: float, seed: Optional[int]) -> None:
        rng = np.random.default_rng(seed)
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            self.weights.append(rng.uniform(lower, upper, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))
            if self.use_rmsnorm:
                self.rms_gamma.append(np.ones((1, fan_out)))
                self.grad_rms_gamma.append(np.zeros((1, fan_out)))

    def _init_normal(self, mean: float, variance: float, seed: Optional[int]) -> None:

        std = np.sqrt(variance)
        rng = np.random.default_rng(seed)
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            self.weights.append(rng.normal(mean, std, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))
            if self.use_rmsnorm:
                self.rms_gamma.append(np.ones((1, fan_out)))
                self.grad_rms_gamma.append(np.zeros((1, fan_out)))

    def _init_xavier(self, seed: Optional[int]) -> None:
        rng = np.random.default_rng(seed)
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))
            if self.use_rmsnorm:
                self.rms_gamma.append(np.ones((1, fan_out)))
                self.grad_rms_gamma.append(np.zeros((1, fan_out)))

    def _init_he(self, seed: Optional[int]) -> None:
        rng = np.random.default_rng(seed)
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))
            if self.use_rmsnorm:
                self.rms_gamma.append(np.ones((1, fan_out)))
                self.grad_rms_gamma.append(np.zeros((1, fan_out)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Menyimpan Z (hasil pre-activation, kombinasi linear) dan
        # A (hasil post-activation) untuk setiap layer, termasuk input sebagai a[0].
        self.z_list: List[np.ndarray] = []
        self.z_pre_norm_list: List[np.ndarray] = []
        self.a_list: List[np.ndarray] = []

        a = X
        self.a_list.append(a)

        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_pre_norm_list.append(z)
            if self.use_rmsnorm:
                z = rmsnorm_forward(z, self.rms_gamma[i], eps=self.rmsnorm_eps)
            self.z_list.append(z)

            activation_fn = get_activation(self.activations[i])
            a = activation_fn(z)
            self.a_list.append(a)

        return a

    def backward(self, y_true, loss_type="mse", reg_type=None, lambda_=0.0):
        y_pred = self.a_list[-1]

        # Get dL/da for output layer based on loss type
        if loss_type == "mse":
            dL_da = MSE.backward(y_true, y_pred)
        elif loss_type == "binary_crossentropy":
            dL_da = BinaryCrossEntropy.backward(y_true, y_pred)
        elif loss_type == "categorical_crossentropy":
            dL_da = CategoricalCrossEntropy.backward(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Compute regularization gradients
        reg_grads = compute_regularization_gradient(self.weights, reg_type, lambda_)

        # Backprop through layers (from last to first)
        for i in reversed(range(self.num_layers - 1)):
            z = self.z_list[i]
            z_pre = self.z_pre_norm_list[i]
            a_prev = self.a_list[i]
            activation_name = self.activations[i]

            # Compute delta = dL/dz
            if activation_name == "softmax":
                delta = softmax_derivative_vectorized(z, dL_da)
            else:
                activation_deriv = get_activation_derivative(activation_name)
                delta = dL_da * activation_deriv(z)

            # If RMSNorm enabled, backprop before weight/bias gradients
            if self.use_rmsnorm:
                delta, dgamma = rmsnorm_backward(
                    delta, z_pre, self.rms_gamma[i], eps=self.rmsnorm_eps
                )
                self.grad_rms_gamma[i] = dgamma

            # Compute gradients for weights and biases (add regularization to weights)
            self.grad_weights[i] = a_prev.T @ delta + reg_grads[i]
            self.grad_biases[i] = np.sum(delta, axis=0, keepdims=True)

            # Propagate gradient to previous layer
            if i > 0:
                dL_da = delta @ self.weights[i].T

        return self.grad_weights, self.grad_biases

    def compute_loss(self, y_true, y_pred, loss_type="mse", reg_type=None, lambda_=0.0):
        if loss_type == "mse":
            data_loss = MSE.forward(y_true, y_pred)
        elif loss_type == "binary_crossentropy":
            data_loss = BinaryCrossEntropy.forward(y_true, y_pred)
        elif loss_type == "categorical_crossentropy":
            data_loss = CategoricalCrossEntropy.forward(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        reg_loss = compute_regularization_loss(self.weights, reg_type, lambda_)
        return data_loss + reg_loss

    def _sgd_step(self, learning_rate: float) -> None:
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * self.grad_weights[i]
            self.biases[i] -= learning_rate * self.grad_biases[i]
            if self.use_rmsnorm:
                self.rms_gamma[i] -= learning_rate * self.grad_rms_gamma[i]

    def fit(
        self,
        X,
        y,
        X_val,
        y_val,
        batch_size,
        learning_rate,
        epochs,
        verbose,
        loss_type="mse",
        reg_type=None,
        lambda_=0.0,
    ):
        X = np.asarray(X)
        y = np.asarray(y)
        X_val = None if X_val is None else np.asarray(X_val)
        y_val = None if y_val is None else np.asarray(y_val)

        if X.shape[0] != y.shape[0]:
            raise ValueError("Jumlah sample X dan y harus sama.")
        if X_val is not None and y_val is not None and X_val.shape[0] != y_val.shape[0]:
            raise ValueError("Jumlah sample X_val dan y_val harus sama.")
        if batch_size <= 0:
            raise ValueError("batch_size harus > 0.")
        if epochs <= 0:
            raise ValueError("epochs harus > 0.")
        if verbose not in (0, 1):
            raise ValueError("verbose hanya boleh 0 atau 1.")

        num_samples = X.shape[0]
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                y_pred_batch = self.forward(X_batch)
                self.backward(
                    y_true=y_batch,
                    loss_type=loss_type,
                    reg_type=reg_type,
                    lambda_=lambda_,
                )
                self._sgd_step(learning_rate)

            train_pred = self.forward(X)
            train_loss = self.compute_loss(
                y_true=y,
                y_pred=train_pred,
                loss_type=loss_type,
                reg_type=reg_type,
                lambda_=lambda_,
            )

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(
                    y_true=y_val,
                    y_pred=val_pred,
                    loss_type=loss_type,
                    reg_type=reg_type,
                    lambda_=lambda_,
                )
            else:
                val_loss = np.nan

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))

            if verbose == 1:
                progress = (epoch + 1) / epochs
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "#" * filled + "-" * (bar_len - filled)
                if np.isnan(val_loss):
                    msg = (
                        f"\rEpoch {epoch + 1}/{epochs} [{bar}] "
                        f"train_loss={train_loss:.6f}"
                    )
                else:
                    msg = (
                        f"\rEpoch {epoch + 1}/{epochs} [{bar}] "
                        f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
                    )
                print(msg, end="", flush=True)

        if verbose == 1:
            print()

        return history

    def plot_weight_distribution(self, layers: Optional[List[int]] = None) -> None:

        # plotting histogram bobot dan bias dari setiap layer
        if layers is None:
            layers = list(range(len(self.weights)))

        num_layers = len(layers)
        fig, axes = plt.subplots(
            num_layers, 2, figsize=(12, 4 * num_layers), squeeze=False
        )
        fig.suptitle("Distribusi Bobot dan Bias", fontsize=14)

        for row, idx in enumerate(layers):
            axes[row, 0].hist(self.weights[idx].flatten(), bins=30, edgecolor="black")
            axes[row, 0].set_title(
                f"Layer {idx + 1} Bobot ({self.layer_sizes[idx]}->{self.layer_sizes[idx + 1]})"
            )
            axes[row, 0].set_xlabel("Nilai")
            axes[row, 0].set_ylabel("Frekuensi")

            axes[row, 1].hist(
                self.biases[idx].flatten(), bins=30, edgecolor="black", color="orange"
            )
            axes[row, 1].set_title(f"Layer {idx + 1} Bias")
            axes[row, 1].set_xlabel("Nilai")
            axes[row, 1].set_ylabel("Frekuensi")

        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layers: Optional[List[int]] = None) -> None:

        # Plotting histogram gradien bobot dan bias dari setiap layer yang dipilih.
        if not hasattr(self, "grad_weights") or not self.grad_weights:
            raise RuntimeError(
                "Gradien belum tersedia. Jalankan backward pass terlebih dahulu."
            )

        if layers is None:
            layers = list(range(len(self.grad_weights)))

        num_layers = len(layers)
        fig, axes = plt.subplots(
            num_layers, 2, figsize=(12, 4 * num_layers), squeeze=False
        )
        fig.suptitle("Distribusi Gradien", fontsize=14)

        for row, idx in enumerate(layers):
            axes[row, 0].hist(
                self.grad_weights[idx].flatten(), bins=30, edgecolor="black"
            )
            axes[row, 0].set_title(f"Layer {idx + 1} Gradien Bobot")
            axes[row, 0].set_xlabel("Nilai")
            axes[row, 0].set_ylabel("Frekuensi")

            axes[row, 1].hist(
                self.grad_biases[idx].flatten(),
                bins=30,
                edgecolor="black",
                color="orange",
            )
            axes[row, 1].set_title(f"Layer {idx + 1} Gradien Bias")
            axes[row, 1].set_xlabel("Nilai")
            axes[row, 1].set_ylabel("Frekuensi")

        plt.tight_layout()
        plt.show()

    def print_architecture(self) -> str:
        lines = ["FFNN Summary", "=" * 50]
        lines.append(f"Input size : {self.layer_sizes[0]}")
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            num_params = fan_in * fan_out + fan_out
            lines.append(
                f"Layer {i + 1:>2d}   : {fan_in} -> {fan_out}  "
                f"activation={self.activations[i]:<8s}  params={num_params}"
            )
        total = sum(
            self.layer_sizes[i] * self.layer_sizes[i + 1] + self.layer_sizes[i + 1]
            for i in range(self.num_layers - 1)
        )
        lines.append("=" * 50)
        lines.append(f"Total parameters: {total}")
        return "\n".join(lines)

    def save(self, path: str) -> None:

        # Menyimpan arsitektur dan bobot model ke file .npz
        save_dict = {
            "layer_sizes": np.array(self.layer_sizes),
            "activations": np.array(self.activations),
            "use_rmsnorm": np.array(self.use_rmsnorm),
            "rmsnorm_eps": np.array(self.rmsnorm_eps),
        }
        for i, w in enumerate(self.weights):
            save_dict[f"W_{i}"] = w
        for i, b in enumerate(self.biases):
            save_dict[f"b_{i}"] = b
        if self.use_rmsnorm:
            for i, g in enumerate(self.rms_gamma):
                save_dict[f"g_{i}"] = g

        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "FFNN":

        # Memuat arsitektur dan bobot model dari file .npz
        data = np.load(path, allow_pickle=True)
        layer_sizes = data["layer_sizes"].tolist()
        activations = data["activations"].tolist()
        use_rmsnorm = bool(data.get("use_rmsnorm", False))
        rmsnorm_eps = float(data.get("rmsnorm_eps", 1e-8))

        model = cls(
            layer_numbers_list=layer_sizes,
            activation_function_list=activations,
            use_rmsnorm=use_rmsnorm,
            rmsnorm_eps=rmsnorm_eps,
        )

        num_layers = len(layer_sizes)
        model.weights = [data[f"W_{i}"].copy() for i in range(num_layers - 1)]
        model.biases = [data[f"b_{i}"].copy() for i in range(num_layers - 1)]
        if model.use_rmsnorm:
            model.rms_gamma = [data[f"g_{i}"].copy() for i in range(num_layers - 1)]
            model.grad_rms_gamma = [np.zeros_like(g) for g in model.rms_gamma]

        return model
