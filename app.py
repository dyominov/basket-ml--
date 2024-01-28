from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Загрузка обученной модели и других данных
model = joblib.load('stacking_regressor_model.pkl')
encoder = joblib.load('team_encoder.pkl')
average_score_away = joblib.load('average_score_away.pkl')
average_score_home = joblib.load('average_score_home.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    away_team = data['awayTeam']
    home_team = data['homeTeam']
    
    encoded_teams = encoder.transform([[away_team, home_team]])

    features = pd.DataFrame(encoded_teams, columns=encoder.get_feature_names_out())

    average_away = average_score_away.get(away_team, np.nan)
    average_home = average_score_home.get(home_team, np.nan)

    features['averageScoreAway'] = average_away
    features['averageScoreHome'] = average_home

    prediction_sklearn = model.predict(features)

    return jsonify({
        'predictedTotalScoreSklearn': prediction_sklearn[0].item() if np.isscalar(prediction_sklearn[0]) else prediction_sklearn[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
