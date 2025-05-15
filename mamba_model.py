import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import optuna
import random
import os
import json

class MambaBlock(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, dropout_rate, kernel_size):
        super(MambaBlock, self).__init__()
        self.input_proj = nn.Linear(input_dim, state_dim)
        self.conv_layer = nn.Conv1d(state_dim, state_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.state_a = nn.Parameter(torch.randn(state_dim, state_dim))
        self.state_b = nn.Parameter(torch.randn(state_dim, state_dim))
        self.state_c = nn.Parameter(torch.randn(state_dim, output_dim))

        self.activation = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(state_dim)

    def forward(self, x, output_length):
        u = self.input_proj(x)
        v = self.activation(u)
        B, T, D = v.size()

        v = v.permute(0, 2, 1)
        v = self.conv_layer(v)
        v = self.batch_norm(v)
        v = v.permute(0, 2, 1)

        s = torch.zeros(B, D, device=x.device)
        y = []
        for t in range(output_length):
            s = s @ self.state_a + v[:, t, :] @ self.state_b
            s = self.norm(s)
            s = self.dropout(s)
            y_t = s @ self.state_c
            y.append(y_t)

        return torch.stack(y, dim=1)

class MambaModel(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, dropout_rate, kernel_size):
        super(MambaModel, self).__init__()
        self.mamba_block = MambaBlock(input_dim, state_dim, output_dim, dropout_rate, kernel_size)

    def forward(self, x, output_length):
        return self.mamba_block(x, output_length)


def prepare_data(data, input_seq_length, output_seq_length, batch_size, target_features):
    data = pd.read_csv(data)
    data = data.select_dtypes(include=['int', 'float'])
    data.fillna(0, inplace=True)

    input_features = data.columns.tolist()

    target_data = data[target_features]
    input_data = data[input_features]

    scaler_input = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_input_data = scaler_input.fit_transform(input_data)
    scaled_target_data = scaler_target.fit_transform(target_data)

    input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32)
    target_tensor = torch.tensor(scaled_target_data, dtype=torch.float32)

    X, Y = create_sequences(input_tensor, target_tensor, input_seq_length, output_seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X, Y, scaler_target


def create_sequences(data, target, input_seq_length, output_seq_length):
    X, Y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length):
        X.append(data[i:i + input_seq_length, :])
        Y.append(target[i + input_seq_length:i + input_seq_length + output_seq_length, :])
    return torch.stack(X), torch.stack(Y)


def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    counter = 0
    patience = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch, y_batch.shape[1])
            loss = mse_loss(output, y_batch)
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch}, Loss: {train_losses[-1]:.4f}")

        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_targets = []
            val_loss = 0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch, y_batch.shape[1])

                loss = mse_loss(predictions, y_batch)
                val_loss += loss.item()

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
            val_losses.append(val_loss / len(val_loader))
            print(f"Validation Loss: {val_losses[-1]:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    return model, train_losses, val_losses, all_targets, all_predictions


def evaluate_model(model, test_loader, device):
    mse_loss = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        test_loss = 0
        all_predictions = []
        all_targets = []
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch, y_batch.shape[1])

            loss = mse_loss(predictions, y_batch)
            test_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss, all_predictions, all_targets


def visualisation_training(train_losses, val_losses, all_predictions, all_targets, scaler_target):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if len(all_predictions) == 0 or len(all_targets) == 0:
        print("No prediction data available for plotting.")
        return

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    all_predictions_flat = all_predictions.reshape(-1, all_predictions.shape[-1])
    all_targets_flat = all_targets.reshape(-1, all_targets.shape[-1])

    all_predictions_rescaled = scaler_target.inverse_transform(all_predictions_flat)
    all_targets_rescaled = scaler_target.inverse_transform(all_targets_flat)

    plt.figure(figsize=(15, 8))
    plt.plot(all_targets_rescaled[:, 9], label="Actual", color="green")
    plt.plot(all_predictions_rescaled[:, 9], label="Predicted", linestyle="--", color="orange")
    plt.ylabel("Temperature")
    plt.xlabel("Time Step (flattened)")
    plt.title("Predicted vs Actual (All Training Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.plot(all_targets_rescaled[:, 0], label="Actual", color="green")
    plt.plot(all_predictions_rescaled[:, 0], label="Predicted", linestyle="--", color="orange")
    plt.ylabel("Stink")
    plt.xlabel("Time Step (flattened)")
    plt.title("Predicted vs Actual (All Training Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.plot(all_targets_rescaled[:, 2], label="Actual", color="green")
    plt.plot(all_predictions_rescaled[:, 2], label="Predicted", linestyle="--", color="orange")
    plt.ylabel("PM10")
    plt.xlabel("Time Step (flattened)")
    plt.title("Predicted vs Actual (All Training Samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualisation_test(all_predictions, all_targets, scaler_target, output_seq_len):
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    num_features = all_predictions.shape[-1]

    all_predictions_2d = all_predictions.reshape(-1, num_features)
    all_targets_2d = all_targets.reshape(-1, num_features)

    all_predictions_rescaled = scaler_target.inverse_transform(all_predictions_2d)
    all_targets_rescaled = scaler_target.inverse_transform(all_targets_2d)

    all_predictions_reshaped = all_predictions_rescaled.reshape(-1, output_seq_len, num_features)
    all_targets_reshaped = all_targets_rescaled.reshape(-1, output_seq_len, num_features)

    num_samples_to_plot = min(3, len(all_predictions_reshaped))

    plt.figure(figsize=(12, 4 * num_samples_to_plot))

    for i in range(num_samples_to_plot):
        plt.subplot(num_samples_to_plot, 1, i + 1)
        plt.plot(all_targets_reshaped[i, :, 0], label="Actual", color="green")
        plt.plot(all_predictions_reshaped[i, :, 0], label="Predicted", linestyle="--", color="orange")
        plt.ylabel("Stink")
        plt.title(f"Test Sample {i + 1}")
        plt.grid(True)
        plt.legend()

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()

    for i in range(num_samples_to_plot):
        plt.subplot(num_samples_to_plot, 1, i + 1)
        plt.plot(all_targets_reshaped[i, :, 9], label="Actual", color="green")
        plt.plot(all_predictions_reshaped[i, :, 9], label="Predicted", linestyle="--", color="orange")
        plt.ylabel("Temperature")
        plt.title(f"Test Sample {i + 1}")
        plt.grid(True)
        plt.legend()

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()

    for i in range(num_samples_to_plot):
        plt.subplot(num_samples_to_plot, 1, i + 1)
        plt.plot(all_targets_reshaped[i, :, 2], label="Actual", color="green")
        plt.plot(all_predictions_reshaped[i, :, 2], label="Predicted", linestyle="--", color="orange")
        plt.ylabel("PM10")
        plt.title(f"Test Sample {i + 1}")
        plt.grid(True)
        plt.legend()

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = trial.suggest_categorical('state_dim', [32, 64, 128, 256, 512,1024])
    learning_rate = trial.suggest_categorical('learning_rate', [0.00001, 0.0001, 0.001, 0.01])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    input_seq_length = trial.suggest_categorical('input_seq_length', [7,14,21])

    output_seq_length = 2

    data = "weather_data_two.csv"
    target_features = [
        "stink","no2","pm10","pm25","so2","o3","cloud_cover","wind_direction","wind_speed","temperature","humidity","pressure","precipitation"
    ]
    train_loader, val_loader, test_loader, X, Y, scaler_target = prepare_data(
        data, input_seq_length, output_seq_length, batch_size, target_features
    )

    model = MambaModel(
        input_dim=X.shape[2],
        state_dim=state_dim,
        output_dim=Y.shape[2],
        dropout_rate=dropout,
        kernel_size=kernel_size
    )

    _, _, val_losses, _,_ = train_model(model, train_loader, val_loader, num_epochs=15, device=device,
                                   learning_rate=learning_rate)
    test_loss, _, _ = evaluate_model(model, test_loader, device)

    return test_loss

def save_best_params(best_params, filename="best_params_m1.json"):
    with open(filename, 'w') as f:
        json.dump(best_params, f)

def load_best_params(filename="best_params_m1.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = "weather_data_test.csv"
    target_features = [
        "stink","no2","pm10","pm25","so2","o3","cloud_cover","wind_direction","wind_speed","temperature","humidity","pressure","precipitation"
    ]

    best_params = load_best_params()

    if best_params is None:
        study = optuna.create_study(direction="minimize", study_name="weather_study")
        study.optimize(objective, n_trials=100)

        print(f"Best hyperparameters: {study.best_params}")

        best_params = study.best_params
        save_best_params(best_params)


    best_state_dim = best_params["state_dim"]
    best_dropout = best_params["dropout"]
    best_kernel = best_params["kernel_size"]
    best_input_seq_length = best_params["input_seq_length"]
    best_batch_size = best_params["batch_size"]
    best_learning_rate = best_params["learning_rate"]

    output_seq_length = 7

    train_loader, val_loader, test_loader, X, Y, scaler_target = prepare_data(data,best_input_seq_length,output_seq_length,best_batch_size,target_features)

    model = MambaModel(input_dim=X.shape[2], state_dim=best_state_dim, output_dim=Y.shape[2],dropout_rate=best_dropout,kernel_size=best_kernel).to(device)

    trained_model, train_losses, val_losses, all_targets, all_predictions = train_model(model, train_loader, val_loader, num_epochs=50, device=device, learning_rate=best_learning_rate)
    test_loss, all_predictions_t, all_targets_t = evaluate_model(trained_model, test_loader, device)
    visualisation_training(train_losses, val_losses, all_predictions, all_targets, scaler_target)
    visualisation_test(all_predictions_t, all_targets_t, scaler_target,output_seq_length)

if __name__ == '__main__':
    main()
