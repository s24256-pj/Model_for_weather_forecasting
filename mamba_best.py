import torch
import pandas as pd
import numpy as np
import joblib
import json
from mamba_model import MambaModel,MambaBlock
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

csv_path = "weather_data_two.csv"
model_path = "mamba_model_weights.pth"
params_path = "model_m_params.json"
scaler_input_path = "scaler_input_m.save"
scaler_target_path = "scaler_target_m.save"

with open(params_path, "r") as f:
    params = json.load(f)

scaler_input = joblib.load(scaler_input_path)
scaler_target = joblib.load(scaler_target_path)

data = pd.read_csv(csv_path)

target_columns = ["stink","no2","pm10","pm25","so2","o3","cloud_cover","wind_direction","wind_speed","temperature","humidity","pressure","precipitation"]

data["date"] = pd.to_datetime(data["date"])
last_date = data["date"].iloc[-1]

data = data.select_dtypes(include=['int', 'float'])
data.fillna(0, inplace=True)

input_columns = data.columns.tolist()
input_data = data[input_columns]
target_day_index = -7  # 7 dni przed końcem
input_seq_length = seq_len= 7

print(input_seq_length)
start_idx = target_day_index - input_seq_length
end_idx = target_day_index

last_sequence = data[input_columns].iloc[start_idx:end_idx].copy()

if last_sequence.shape[0] != input_seq_length:
    raise ValueError(f"Potrzebujesz co najmniej {input_seq_length} danych do predykcji.")

seq = data[input_columns].iloc[start_idx:end_idx].copy()  # shape (21, 17)

scaled_part = scaler_input.transform(seq[input_columns])  # -> (21, len(features_to_scale))

input_tensor = torch.tensor(scaled_part, dtype=torch.float32).unsqueeze(0)

model = MambaModel(**params)
model.load_state_dict(torch.load(model_path))
model.eval()

num_features = len(target_columns)

with torch.no_grad():
    output = model(input_tensor, output_length=7)

output_np = output.squeeze(0).numpy()
predicted = scaler_target.inverse_transform(output_np)

predicted_dates = [last_date + timedelta(days=i+1) for i in range(predicted.shape[0])]

print("Prognoza:")
for i, (date, row) in enumerate(zip(predicted_dates, predicted)):
    print(f"\n{date.strftime('%Y-%m-%d')} (Dzień {i+1}):")
    for name, value in zip(target_columns, row):
        print(f"  {name:<15}: {value:.3f}")