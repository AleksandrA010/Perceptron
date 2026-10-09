import numpy as np
import pandas as pd
import argparse
import pickle

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

def predict_forward(X, layers_weights):
    out = X
    for layer in layers_weights:
        W = layer['W']
        b = layer['b']
        act = layer['activation']
        
        z = np.dot(out, W) + b
        if act == 'sigmoid':
            out = sigmoid(z)
        elif act == 'relu':
            out = relu(z)
        elif act == 'softmax':
            out = softmax(z)
        else:
            out = z
    return out

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MLP model")
    parser.add_argument("--model", type=str, default="saved_model.npy", help="Path to saved model weights")
    parser.add_argument("--dataset", type=str, default="data_valid.csv", help="Dataset to evaluate on")
    args = parser.parse_args()

    # Загрузка весов
    with open(args.model, 'rb') as f:
        layers_weights = pickle.load(f)

    # Загрузка данных
    df = pd.read_csv(args.dataset)
    X = df.drop(columns=['diagnosis']).values
    y = df['diagnosis'].values
    y_oh = np.eye(2)[y]

    # Прямой проход
    probs = predict_forward(X, layers_weights)
    loss = categorical_cross_entropy(y_oh, probs)
    
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)

    print(f"--- Evaluation Results on '{args.dataset}' ---")
    print(f"Binary Cross-Entropy Loss: {loss:.4f}")
    print(f"Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()