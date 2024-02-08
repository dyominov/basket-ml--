from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


# Загрузка обученной модели
model = joblib.load('stacking_regressor_model.pkl')
encoder = joblib.load('team_encoder.pkl')
encoder_torch = joblib.load('team_encoder_torch.pkl')

average_score_away = joblib.load('average_score_away.pkl')
average_score_home = joblib.load('average_score_home.pkl')

class RegressionModel(nn.Module):
    def __init__(self, input_size=1294):  # Измените здесь
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size_correct = len(encoder_torch.get_feature_names_out()) + 2

model_torch = RegressionModel(input_size=input_size_correct)
model_torch.load_state_dict(torch.load('simple_nn_model.pth'))
model_torch.eval()

def fetch_matches():
    url = "https://api.sportify.bet/echo/v1/events?sport=e-basketball&competition=north-america-2k-basketball-regular-cup&key=market_type&lang=en&bookmaker=netbetcom&id=6ece20c6-b62f-41e5-bd74-39fcbbdea4d9"
    response = requests.get(url)
    matches = []

    if response.status_code == 200:
        data = response.json()
        for event in data['tree'][0]['competitions'][0]['events']:
            if len(event['competitors']) == 2:
                matches.append((event['competitors'][0]['name'], event['competitors'][1]['name']))
    return matches

@app.route('/predict', methods=['GET'])
def predict():
    matches = fetch_matches()
    predictions = []

    for home_team, away_team in matches:
        encoded_teams = encoder.transform([[away_team, home_team]])
        encoded_teams_torch = encoder_torch.transform([[away_team, home_team]])

        features = pd.DataFrame(encoded_teams, columns=encoder.get_feature_names_out())
        features_torch = pd.DataFrame(encoded_teams_torch, columns=encoder_torch.get_feature_names_out())

        average_away = average_score_away.get(away_team, np.nan)
        average_home = average_score_home.get(home_team, np.nan)

        features['averageScoreAway'] = average_away
        features['averageScoreHome'] = average_home
        features_torch['averageScoreAway'] = average_away
        features_torch['averageScoreHome'] = average_home

        # Проверка на NaN в features
        if features.isna().any().any():
            continue  # Пропускаем эту итерацию, если есть NaN

        prediction_sklearn = model.predict(features)

        # Масштабирование данных для модели PyTorch
        scaler = joblib.load('scaler_model_torch.pkl')
        scaled_features_torch = scaler.transform(features_torch)
        features_tensor = torch.tensor(scaled_features_torch.astype(np.float32))

        with torch.no_grad():
            prediction_torch = model_torch(features_tensor).numpy()

        predictions.append({
            'match': f"{home_team} vs {away_team}",
            'predictedTotalScoreSklearn': prediction_sklearn[0].item() if np.isscalar(prediction_sklearn[0]) else prediction_sklearn[0].tolist(),
            'predictedTotalScoreTorch': prediction_torch[0][0].item()
        })

    return jsonify(predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
