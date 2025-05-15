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


class WeatherModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, seq_len, output_features, dropout, dim_feedforward_encoder, dim_feedforward_decoder):
        super(WeatherModel, self).__init__()

        self.input_linear_mapping = nn.Linear(input_dim, embed_dim)
        self.target_linear_mapping = nn.Linear(output_features, embed_dim)
        self.input_bn = nn.BatchNorm1d(embed_dim)
        self.output_bn = nn.BatchNorm1d(embed_dim)

        self.position_encoder = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward_encoder, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward_decoder, dropout=dropout,
                                                        batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        self.output_layer = nn.Linear(embed_dim, output_features)

    def forward(self, src, tgt):
        src_pos = self.position_encoder[:, :src.shape[1], :].expand(src.shape[0], -1, -1)
        src_emb = self.input_linear_mapping(src) + src_pos
        src_emb = self.norm(src_emb)
        src_emb = src_emb.permute(0, 2, 1)
        src_emb = self.input_bn(src_emb)
        src_emb = src_emb.permute(0, 2, 1)
        src_emb = self.dropout(src_emb)

        encoder_output = self.transformer_encoder(src_emb)

        tgt_pos = self.position_encoder[:, :tgt.shape[1], :].expand(tgt.shape[0], -1, -1)
        tgt_emb = self.target_linear_mapping(tgt) + tgt_pos
        tgt_emb = self.norm(tgt_emb)
        tgt_emb = tgt_emb.permute(0, 2, 1) # -> [B,E,S]
        tgt_emb = self.output_bn(tgt_emb)
        tgt_emb = tgt_emb.permute(0, 2, 1) # -> [B,S,E]
        tgt_emb = self.dropout(tgt_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        decoder_output = self.transformer_decoder(tgt_emb, encoder_output, tgt_mask=tgt_mask)

        return self.output_layer(decoder_output)

def prepare_data(data, input_seq_length, output_seq_length, batch_size,target_features):
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

    return train_loader, val_loader, test_loader, X, Y

def create_sequences(data, target, input_seq_length, output_seq_length):
    X, Y = [], []
    for i in range(len(data) - input_seq_length - output_seq_length):
        X.append(data[i:i + input_seq_length, :])
        Y.append(target[i + input_seq_length:i + input_seq_length + output_seq_length, :])
    return torch.stack(X), torch.stack(Y)

def train_model(model, train_loader, val_loader, num_epochs,device, learning_rate):
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    counter = 0
    patience = 3

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            tgt_input = torch.zeros((X_batch.size(0), 1, y_batch.size(2))).to(device)
            output = model(X_batch, tgt_input)
            loss = mse_loss(output, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        print(f"Epoch {epoch}, Loss: {train_losses[-1]:.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0

            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                tgt_input = y_batch[:, :-1, :]
                predictions = model(X_batch, tgt_input)
                loss = mse_loss(predictions, y_batch[:, 1:, :])
                val_loss += loss.item()

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

    return model, train_losses, val_losses

def evaluate_model(model, test_loader, device):
    mse_loss = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        test_loss = 0
        all_predictions = []
        all_targets = []

        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            tgt_input = y_batch[:, :-1, :]
            predictions = model(X_batch, tgt_input)
            loss = mse_loss(predictions, y_batch[:, 1:, :])
            test_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch[:, 1:, :].cpu().numpy())

        print(f"Test Loss: {test_loss:.4f}")
        return test_loss, all_predictions, all_targets

def visualisation(train_losses, val_losses, all_predictions, all_targets):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.show()

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    num_samples = 5

    plt.figure(figsize=(12, 7))

    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(all_targets[:, 0, 2], label="Actual", color="green")
        plt.plot(all_predictions[:, 0, 2], label="Predicted", linestyle="dashed", color="orange")
        plt.ylabel("Value")
        plt.legend()
        plt.title(f"Sample {i + 1}")

    plt.xlabel("Time Step")
    plt.show()

def set_seed(seed=42):
    seed = 42
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

    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    embed_dim = trial.suggest_categorical('embed_dim', [32, 64, 128, 256, 512])
    learning_rate = trial.suggest_categorical('learning_rate', [0.00001, 0.0001, 0.001, 0.01])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    dim_feedforward_encoder = trial.suggest_categorical('dim_feedforward_encoder', [32, 64, 128, 256, 512, 1024, 2048])
    dim_feedforward_decoder = trial.suggest_categorical('dim_feedforward_decoder', [32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical('batch_size', [32, 64])

    input_seq_length = 7
    output_seq_length = 7

    data = "weather_data_full.csv"
    target_features = [
        "Average Daily Total Cloud Cover [oktas]",
        "Average Daily Wind Speed [m/s]",
        "Average Daily Temperature [°C]",
        "Average Daily Relative Humidity [%]",
        "Average Daily Station-Level Pressure [hPa]",
        "Daily Precipitation Sum [mm]",
        "Night Precipitation Sum [mm]"
    ]
    train_loader, val_loader, test_loader, X, Y = prepare_data(
        data, input_seq_length, output_seq_length, batch_size, target_features
    )

    model = WeatherModel(
        input_dim=X.shape[2],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=input_seq_length,
        output_features=Y.shape[2],
        dropout=dropout,
        dim_feedforward_encoder=dim_feedforward_encoder,
        dim_feedforward_decoder=dim_feedforward_decoder
    )

    _, _, val_losses = train_model(model, train_loader, val_loader, num_epochs=15, device=device,
                                   learning_rate=learning_rate)
    return val_losses[-1]

def save_best_params(best_params, filename="best_params.json"):
    with open(filename, 'w') as f:
        json.dump(best_params, f)

def load_best_params(filename="best_params.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_seq_length = 7
    output_seq_length = 7

    data = "weather_data_full.csv"
    target_features = [
        "Average Daily Total Cloud Cover [oktas]",
        "Average Daily Wind Speed [m/s]",
        "Average Daily Temperature [°C]",
        "Average Daily Relative Humidity [%]",
        "Average Daily Station-Level Pressure [hPa]",
        "Daily Precipitation Sum [mm]",
        "Night Precipitation Sum [mm]"
    ]

    best_params = load_best_params()

    if best_params is None:
        study = optuna.create_study(direction="minimize", study_name="weather_study")
        study.optimize(objective, n_trials=200)

        print(f"Best hyperparameters: {study.best_params}")

        best_params = study.best_params
        save_best_params(best_params)

    b_num_heads = best_params['num_heads']
    b_num_layers = best_params['num_layers']
    b_embed_dim = best_params['embed_dim']
    b_learning_rate = best_params['learning_rate']
    b_dropout = best_params['dropout']
    b_dim_feedforward_encoder = best_params['dim_feedforward_encoder']
    b_dim_feedforward_decoder = best_params['dim_feedforward_decoder']
    b_batch_size = best_params['batch_size']

    train_loader, val_loader, test_loader, X, Y = prepare_data(data,input_seq_length,output_seq_length,b_batch_size,target_features)

    model = WeatherModel(input_dim=X.shape[2], embed_dim=b_embed_dim, num_heads=b_num_heads, num_layers=b_num_layers,
                             seq_len=input_seq_length,
                             output_features=Y.shape[2], dropout=b_dropout, dim_feedforward_encoder=b_dim_feedforward_encoder,
                             dim_feedforward_decoder=b_dim_feedforward_decoder)


    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=30, device=device, learning_rate=b_learning_rate)
    test_loss, all_predictions, all_targets = evaluate_model(trained_model, test_loader, device)
    visualisation(train_losses, val_losses, all_predictions, all_targets)

    torch.save(trained_model, "best_model_full.pth")

if __name__ == "__main__":
    main()
