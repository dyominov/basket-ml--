from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    away_team = data['awayTeam']
    home_team = data['homeTeam']
    
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

    prediction_sklearn = model.predict(features)

     # Масштабирование данных для модели PyTorch
    scaler = joblib.load('scaler_model_torch.pkl')
    scaled_features_torch = scaler.transform(features_torch)
    

    features_tensor = torch.tensor(scaled_features_torch.astype(np.float32))
    
    with torch.no_grad():
        print(features_tensor.shape)  
        prediction_torch = model_torch(features_tensor).numpy()
    
    return jsonify({
        'predictedTotalScoreSklearn': prediction_sklearn[0].item() if np.isscalar(prediction_sklearn[0]) else prediction_sklearn[0].tolist(),
        'predictedTotalScoreTorch': prediction_torch[0][0].item()
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
