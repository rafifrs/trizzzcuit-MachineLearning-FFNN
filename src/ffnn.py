from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from activations import (
    get_activation,
    get_activation_derivative,
    softmax_derivative_vectorized,
)
from losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy


class FFNN:
    VALID_ACTIVATIONS = {"linear", "relu", "sigmoid", "tanh", "softmax"}
    VALID_INIT_METHODS = {"zero", "uniform", "normal"}

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

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.grad_weights: List[np.ndarray] = []
        self.grad_biases: List[np.ndarray] = []

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

    def _init_zero(self) -> None:
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            self.weights.append(np.zeros((fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))

    def _init_uniform(self, lower: float, upper: float, seed: Optional[int]) -> None:
        rng = np.random.default_rng(seed)
        for i in range(self.num_layers - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            self.weights.append(rng.uniform(lower, upper, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))
            self.grad_weights.append(np.zeros((fan_in, fan_out)))
            self.grad_biases.append(np.zeros((1, fan_out)))

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

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Menyimpan Z (hasil pre-activation, kombinasi linear) dan
        # A (hasil post-activation) untuk setiap layer, termasuk input sebagai a[0].
        self.z_list: List[np.ndarray] = []
        self.a_list: List[np.ndarray] = []

        a = X
        self.a_list.append(a)

        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_list.append(z)

            activation_fn = get_activation(self.activations[i])
            a = activation_fn(z)
            self.a_list.append(a)

        return a

    def backward(self, y_true, loss_type="mse"):
        y_pred = self.a_list[-1]

        # Get dL/da for output layer based on loss type
        # Note: loss backward already normalizes by batch size
        if loss_type == "mse":
            dL_da = MSE.backward(y_true, y_pred)
        elif loss_type == "binary_crossentropy":
            dL_da = BinaryCrossEntropy.backward(y_true, y_pred)
        elif loss_type == "categorical_crossentropy":
            dL_da = CategoricalCrossEntropy.backward(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Backprop through layers (from last to first)
        for i in reversed(range(self.num_layers - 1)):
            z = self.z_list[i]
            a_prev = self.a_list[i]
            activation_name = self.activations[i]

            # Compute delta = dL/dz
            if activation_name == "softmax":
                delta = softmax_derivative_vectorized(z, dL_da)
            else:
                activation_deriv = get_activation_derivative(activation_name)
                delta = dL_da * activation_deriv(z)

            # Compute gradients for weights and biases
            self.grad_weights[i] = a_prev.T @ delta
            self.grad_biases[i] = np.sum(delta, axis=0, keepdims=True)

            # Propagate gradient to previous layer
            if i > 0:
                dL_da = delta @ self.weights[i].T

        return self.grad_weights, self.grad_biases

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
        }
        for i, w in enumerate(self.weights):
            save_dict[f"W_{i}"] = w
        for i, b in enumerate(self.biases):
            save_dict[f"b_{i}"] = b

        np.savez(path, **save_dict)

    @classmethod
    def load(cls, path: str) -> "FFNN":

        # Memuat arsitektur dan bobot model dari file .npz
        data = np.load(path, allow_pickle=True)
        layer_sizes = data["layer_sizes"].tolist()
        activations = data["activations"].tolist()

        model = cls(
            layer_numbers_list=layer_sizes, activation_function_list=activations
        )

        num_layers = len(layer_sizes)
        model.weights = [data[f"W_{i}"].copy() for i in range(num_layers - 1)]
        model.biases = [data[f"b_{i}"].copy() for i in range(num_layers - 1)]

        return model
