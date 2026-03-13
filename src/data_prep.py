import numpy as np
import csv
import os

def label_encode(column):
    unique_vals = sorted(list(set(column)))
    val_to_int = {val: i for i, val in enumerate(unique_vals)}
    return np.array([val_to_int[val] for val in column]), val_to_int

def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
        
    data = np.array(data)
    
    categorical_cols = [2, 3, 4, 8, 9]
    numerical_cols = [0, 1, 5, 6, 7, 10]
    target_col = 11
    
    # Encode categorical
    encoded_features = []
    for col_idx in categorical_cols:
        encoded, mapping = label_encode(data[:, col_idx])
        encoded_features.append(encoded)
        
    encoded_features = np.column_stack(encoded_features)
    
    # Parse numerical
    numerical_features = data[:, numerical_cols].astype(float)
    
    # Combine features
    X = np.hstack((numerical_features, encoded_features))
    
    # Encode Target
    y, y_mapping = label_encode(data[:, target_col])
    
    # Normalize features (Standard Scaler)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std
    
    # Train/Val split
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(X.shape[0] * (1 - test_size))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'global_student_placement_and_salary.csv')
    
    X_train, y_train, X_val, y_val = load_and_preprocess_data(data_path)
    print("Data successfully loaded and preprocessed!")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
