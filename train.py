import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle

# ==========================================
# 1. Функции активации и их производные
# ==========================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

# ==========================================
# 2. Класс полносвязного слоя (Dense Layer)
# ==========================================

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='sigmoid', weights_initializer='heUniform'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        
        if weights_initializer == 'heUniform':
            limit = np.sqrt(6.0 / input_dim)
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif weights_initializer == 'xavier':
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))
        
        self.last_input = None
        self.last_z = None
        self.last_a = None
        self.dW = None
        self.db = None

    def forward(self, inputs):
        self.last_input = inputs
        self.last_z = np.dot(inputs, self.W) + self.b
        
        if self.activation_name == 'sigmoid':
            self.last_a = sigmoid(self.last_z)
        elif self.activation_name == 'relu':
            self.last_a = relu(self.last_z)
        elif self.activation_name == 'softmax':
            self.last_a = softmax(self.last_z)
        elif self.activation_name == 'linear':
            self.last_a = self.last_z
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
            
        return self.last_a

    def backward(self, dA_or_dZ):
        m = self.last_input.shape[0]
        
        if self.activation_name == 'softmax':
            dZ = dA_or_dZ
        elif self.activation_name == 'sigmoid':
            dZ = dA_or_dZ * sigmoid_derivative(self.last_a)
        elif self.activation_name == 'relu':
            dZ = dA_or_dZ * relu_derivative(self.last_z)
        else:
            dZ = dA_or_dZ

        self.dW = np.dot(self.last_input.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        
        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev

    def update_params(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

# ==========================================
# 3. Модель Персептрона
# ==========================================

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update_params(lr)

# ==========================================
# 4. Функции потерь и метрик
# ==========================================

def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

def compute_accuracy(y_true, y_pred_probs):
    predictions = np.argmax(y_pred_probs, axis=1)
    targets = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    return np.mean(predictions == targets)

# ==========================================
# 5. Главная функция обучения
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Layer Perceptron from scratch")
    parser.add_argument("--dataset", type=str, default="data_train.csv", help="Training dataset path")
    parser.add_argument("--val_dataset", type=str, default="data_valid.csv", help="Validation dataset path")
    parser.add_argument("--layer", nargs="+", type=int, default=[24, 24, 24], help="Hidden layers neurons")
    parser.add_argument("--epochs", type=int, default=84, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate")
    parser.add_argument("--loss", type=str, default="binaryCrossentropy", help="Loss function")
    parser.add_argument("--model_out", type=str, default="saved_model.npy", help="Path to save trained weights")
    args = parser.parse_args()

    np.random.seed(42)

    # Загрузка данных
    train_df = pd.read_csv(args.dataset)
    valid_df = pd.read_csv(args.val_dataset)

    X_train = train_df.drop(columns=['diagnosis']).values
    y_train = train_df['diagnosis'].values
    
    X_valid = valid_df.drop(columns=['diagnosis']).values
    y_valid = valid_df['diagnosis'].values

    y_train_oh = np.eye(2)[y_train]
    y_valid_oh = np.eye(2)[y_valid]

    print(f"x_train shape: {X_train.shape}")
    print(f"x_valid shape: {X_valid.shape}")

    input_dim = X_train.shape[1]
    output_dim = 2

    net = NeuralNetwork()
    
    prev_dim = input_dim
    for hidden_units in args.layer:
        net.add(DenseLayer(prev_dim, hidden_units, activation='sigmoid', weights_initializer='heUniform'))
        prev_dim = hidden_units
        
    net.add(DenseLayer(prev_dim, output_dim, activation='softmax', weights_initializer='heUniform'))

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    num_samples = X_train.shape[0]

    for epoch in range(1, args.epochs + 1):
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_oh[indices]

        for i in range(0, num_samples, args.batch_size):
            X_batch = X_train_shuffled[i:i+args.batch_size]
            y_batch = y_train_shuffled[i:i+args.batch_size]

            out = net.forward(X_batch)
            dZ_out = out - y_batch
            
            net.backward(dZ_out)
            net.update(args.learning_rate)

        train_preds = net.forward(X_train)
        valid_preds = net.forward(X_valid)

        train_loss = categorical_cross_entropy(y_train_oh, train_preds)
        valid_loss = categorical_cross_entropy(y_valid_oh, valid_preds)

        train_acc = compute_accuracy(y_train, train_preds)
        valid_acc = compute_accuracy(y_valid, valid_preds)

        history['loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['acc'].append(train_acc)
        history['val_acc'].append(valid_acc)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:02d}/{args.epochs:02d} | loss: {train_loss:.4f} - val_loss: {valid_loss:.4f} | acc: {train_acc:.4f} - val_acc: {valid_acc:.4f}")

    weights = [{'W': layer.W, 'b': layer.b, 'activation': layer.activation_name} for layer in net.layers]
    with open(args.model_out, 'wb') as f:
        pickle.dump(weights, f)
    print(f"\n> saving model '{args.model_out}' to disk...")

    # Сохранение двух отдельных файлов
    # 1. Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(history['loss'], label='training loss', color='blue')
    plt.plot(history['val_loss'], label='validation loss', color='orange', linestyle='--')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.close()
    print("Saved 'loss_curve.png'")

    # 2. Accuracy Curve
    plt.figure(figsize=(7, 5))
    plt.plot(history['acc'], label='training acc', color='blue')
    plt.plot(history['val_acc'], label='validation acc', color='orange')
    plt.title('Learning Curves (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.close()
    print("Saved 'accuracy_curve.png'")

if __name__ == "__main__":
    main()