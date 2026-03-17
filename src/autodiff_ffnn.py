from typing import List

from autodiff import Value


def _activate(name: str, v: Value) -> Value:
    name = name.lower()
    if name == "linear":
        return v
    if name == "relu":
        return v.relu()
    if name == "sigmoid":
        return v.sigmoid()
    if name == "tanh":
        return v.tanh()
    raise ValueError(f"Unsupported activation for autodiff: {name}")


def _softmax(values: List[Value]) -> List[Value]:
    max_val = max(v.data for v in values)
    exps = [(v - max_val).exp() for v in values]
    total = Value(0.0)
    for e in exps:
        total = total + e
    return [e / total for e in exps]


class AutodiffFFNN:
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        if len(layer_sizes) < 2:
            raise ValueError("Minimal ada 2 layer (input dan output).")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Jumlah fungsi aktivasi harus sama dengan jumlah layer - 1.")

        self.layer_sizes = layer_sizes
        self.activations = activations

        self.weights: List[List[List[Value]]] = []
        self.biases: List[List[Value]] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            W = [[Value(0.01) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [Value(0.0) for _ in range(fan_out)]
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, x: List[float]) -> List[Value]:
        a = [Value(v) for v in x]
        for layer_idx in range(len(self.weights)):
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            z = []
            for j in range(len(b)):
                s = b[j]
                for i in range(len(a)):
                    s = s + a[i] * W[i][j]
                z.append(s)

            act = self.activations[layer_idx].lower()
            if act == "softmax":
                a = _softmax(z)
            else:
                a = [_activate(act, v) for v in z]
        return a

    def mse_loss(self, y_true: List[float], y_pred: List[Value]) -> Value:
        assert len(y_true) == len(y_pred)
        loss = Value(0.0)
        for i in range(len(y_true)):
            diff = y_pred[i] - y_true[i]
            loss = loss + diff * diff
        return loss * (1.0 / len(y_true))

    def categorical_crossentropy_loss(
        self, y_true: List[float], y_pred: List[Value]
    ) -> Value:
        assert len(y_true) == len(y_pred)
        n = len(y_true)
        loss = Value(0.0)
        for i in range(n):
            loss = loss + (Value(-1.0) * Value(y_true[i]) * y_pred[i].log())
        return loss * (1.0 / n)

    def set_params(self, weights, biases) -> None:
        for i in range(len(self.weights)):
            for r in range(len(self.weights[i])):
                for c in range(len(self.weights[i][r])):
                    self.weights[i][r][c].data = float(weights[i][r][c])
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                self.biases[i][j].data = float(biases[i][j])

    def get_grads(self):
        grad_weights = []
        grad_biases = []
        for W in self.weights:
            grad_W = []
            for row in W:
                grad_W.append([w.grad for w in row])
            grad_weights.append(grad_W)
        for b in self.biases:
            grad_biases.append([v.grad for v in b])
        return grad_weights, grad_biases

    def zero_grad(self) -> None:
        for W in self.weights:
            for row in W:
                for w in row:
                    w.grad = 0.0
        for b in self.biases:
            for v in b:
                v.grad = 0.0

    def sgd_step(self, lr: float) -> None:
        for W in self.weights:
            for row in W:
                for w in row:
                    w.data -= lr * w.grad
        for b in self.biases:
            for v in b:
                v.data -= lr * v.grad
