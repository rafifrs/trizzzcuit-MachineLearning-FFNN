from pathlib import Path

import numpy as np

from data_prep import load_and_preprocess_data
from ffnn import FFNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "global_student_placement_and_salary.csv"
MODEL_PATH = PROJECT_ROOT / "saved_model" / "ffnn_model.npz"


def decode_predictions(y_pred: np.ndarray) -> np.ndarray:
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        return (y_pred >= 0.5).astype(int).reshape(-1)
    return np.argmax(y_pred, axis=1)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model belum ditemukan di {MODEL_PATH}. Jalankan src/save_model.py dulu."
        )

    _, _, X_val, y_val = load_and_preprocess_data(str(DATA_PATH))

    model = FFNN.load(str(MODEL_PATH))
    probs = model.forward(X_val)
    pred = decode_predictions(probs)
    accuracy = float(np.mean(pred == y_val))

    print(f"Model dipakai dari: {MODEL_PATH}")
    print(f"Data dipakai dari: {DATA_PATH}")
    print(f"Akurasi validasi: {accuracy:.4f}")


if __name__ == "__main__":
    main()
