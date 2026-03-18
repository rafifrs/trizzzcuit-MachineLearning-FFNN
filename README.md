
<div align="center">
  <h1>Feedforward Neural Network</h1>
</div>

## Description

Repository ini berisi implementasi Feedforward Neural Network (FFNN) from scratch menggunakan NumPy untuk tugas besar IF3270 Pembelajaran Mesin. Proyek mencakup pipeline lengkap training neural network: inisialisasi bobot, forward propagation, backpropagation, regularisasi, optimizer, normalisasi, utilitas data prep, serta rangkaian notebook eksperimen.


## Features

- Implementasi FFNN multi-layer dari nol (`src/ffnn.py`)
- Activation functions + turunan: `linear`, `relu`, `leaky_relu`, `sigmoid`, `tanh`, `swish`, `softmax`
- Loss functions + turunan: `MSE`, `Binary Cross-Entropy`, `Categorical Cross-Entropy`
- Weight initialization: `zero`, `uniform`, `normal`, `xavier`, `he`
- Regularisasi `L1` dan `L2`
- Optimizer: `SGD` dan `Adam`
- Opsi `RMSNorm` pada layer linear
- Visualisasi distribusi bobot dan gradien per layer
- Save/Load model ke format `.npz`
- Modul autodiff sederhana (`Value`) dan validasi gradien lewat `AutodiffFFNN`
- Data preprocessing untuk dataset placement (`src/data_prep.py`)
- Notebook eksperimen:
  - Aktivasi dan regularisasi
  - Adam optimizer
  - Validasi autodiff
  - Depth/width architecture
  - Learning rate vs sklearn MLP
  - RMSNorm
  - Weight initialization



## Requirements

- Python 3.10+ 
- Library Python:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `jupyter`

## Setup & Installation

1. Clone repository

```bash
git clone https://github.com/rafifrs/trizzzcuit-MachineLearning-FFNN.git
cd trizzzcuit-MachineLearning-FFNN
```

2. Install dependencies

```bash
pip install numpy matplotlib scikit-learn jupyter
```


## Struktur Project

```text
trizzzcuit-MachineLearning-FFNN/
|- data/
|  |- global_student_placement_and_salary.csv
|- doc/
|- src/
|  |- activations.py
|  |- autodiff.py
|  |- autodiff_ffnn.py
|  |- data_prep.py
|  |- ffnn.py
|  |- losses.py
|  |- normalizers.py
|  |- regularizers.py
|  |- notebook/
|     |- eksperimen_activation_regularization.ipynb
|     |- eksperimen_adam_optimizer.ipynb
|     |- eksperimen_autodiff_validasi.ipynb
|     |- eksperimen_depth_width.ipynb
|     |- eksperimen_lr_sklearn.ipynb
|     |- eksperimen_rmsnorm.ipynb
|     |- eksperimen_weight_initialization.ipynb
|- README.md
```

## Pembagian Tugas

| Anggota | Tugas |
|---|---|
| 13523095 | Loss function; data prep & preprocessing; save/load model; weight update; eksperimen learning rate + perbandingan sklearn; bonus RMSNorm; bonus autodiff |
| 13523101 | Fungsi aktivasi; backward propagation; regularisasi; bonus 2 fungsi aktivasi; eksperimen fungsi aktivasi; eksperimen regularisasi; bonus autodiff |
| 13523115 | Kelas FFNN; inisialisasi bobot zero, uniform, normal; forward propagation; metode fit; bonus Xavier dan He; bonus Adam; eksperimen depth-width; eksperimen inisialisasi bobot |