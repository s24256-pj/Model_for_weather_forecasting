import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from weather_model_transformer import WeatherModel
import os
import json

def load_best_params(filename="best_params.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def prepare_sequence_data(data_path, selected_day, input_seq_length):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])

    df.fillna(0, inplace=True)
    df = df.drop(columns="Average Daily Total Cloud Cover [oktas]")
    df_numeric = df.select_dtypes(include=['int', 'float'])

    target_features = [
        #"Average Daily Total Cloud Cover [oktas]",
        "Average Daily Wind Speed [m/s]",
        "Average Daily Temperature [°C]",
        "Average Daily Relative Humidity [%]",
        "Average Daily Station-Level Pressure [hPa]",
        "Daily Precipitation Sum [mm]",
        "Night Precipitation Sum [mm]"
    ]
    input_features = df_numeric.columns.tolist()

    selected_day = pd.to_datetime(selected_day)
    start_date = selected_day - timedelta(days=input_seq_length)

    input_seq = df[(df['Date'] > start_date) & (df['Date'] <= selected_day)].copy()
    input_seq = input_seq.select_dtypes(include=['int', 'float'])

    scaler_input = MinMaxScaler()
    scaled_input_data = scaler_input.fit_transform(input_seq[input_features].values)

    input_tensor = torch.tensor(scaled_input_data, dtype=torch.float32).unsqueeze(0)

    return input_tensor, scaler_input, input_features, target_features, df

def predict_future(model, input_tensor, target_features, output_days):
    model.eval()
    predictions = []
    current_input = input_tensor.clone()

    for days in range(output_days):
        dummy_target = torch.zeros((1, 1, len(target_features)))
        with torch.no_grad():
            pred = model(current_input, dummy_target)
        pred = pred[:, -1:, :]

        predictions.append(pred.squeeze(0).numpy())

        pred_padded = torch.zeros((1, 1, current_input.shape[2]))
        pred_padded[:, :, :pred.shape[2]] = pred
        current_input = torch.cat([current_input[:, 1:, :], pred_padded], dim=1)

    return np.concatenate(predictions, axis=0)

def main():

    selected_day = "2020-08-08"
    data_path = "weather_data_selected.csv"
    input_seq_length = 21
    output_days = 7

    X, scaler_input, input_features, target_features, df = prepare_sequence_data(data_path, selected_day,
                                                                                            input_seq_length)
    best_params = load_best_params()

    if best_params is not None:
        best_params = load_best_params()
    else:
        print("There is no best_params.json file")
        return

    b_num_heads = best_params['num_heads']
    b_num_layers = best_params['num_layers']
    b_embed_dim = best_params['embed_dim']
    b_dropout = best_params['dropout']
    b_dim_feedforward_encoder = best_params['dim_feedforward_encoder']
    b_dim_feedforward_decoder = best_params['dim_feedforward_decoder']

    model = WeatherModel(input_dim=X.shape[2], embed_dim=b_embed_dim, num_heads=b_num_heads, num_layers=b_num_layers,
                         seq_len=input_seq_length,
                         output_features=len(target_features), dropout=b_dropout,
                         dim_feedforward_encoder=b_dim_feedforward_encoder,
                         dim_feedforward_decoder=b_dim_feedforward_decoder)

    weights_path = 'transformer_model_weights.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        model.eval()
    else:
        print(f"There is no weights file: {weights_path}")
        return

    predictions_scaled = predict_future(model, X, target_features, output_days)

    scaler_target = MinMaxScaler()
    scaler_target.fit(df[target_features])
    predictions = scaler_target.inverse_transform(predictions_scaled)

    start_date = pd.to_datetime(selected_day) + timedelta(days=1)
    dates = [start_date + timedelta(days=i) for i in range(output_days)]

    print("\nPredykcje od", selected_day, "na kolejne", output_days, "dni:\n")
    for date, pred in zip(dates, predictions):
        print(f"{date.date()} =>")
        for feature, value in zip(target_features, pred):
            print(f"  {feature}: {value:.2f}")
        print()

if __name__ == "__main__":
    main()