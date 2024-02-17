from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Загружаем модели и OneHotEncoder
model_total = load('model_total.joblib')
model_home = load('model_home.joblib')
model_away = load('model_away.joblib')
model_handicap = load('model_handicap.joblib')
one_hot_encoder = load('one_hot_encoder.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    away_team = data['awayTeam']
    home_team = data['homeTeam']
    
    # Подготовка данных для прогнозирования
    teams_for_prediction = pd.DataFrame({
        'awayTeam': [away_team],
        'homeTeam': [home_team]
    })
    encoded_teams = one_hot_encoder.transform(teams_for_prediction).toarray()
    encoded_teams_df = pd.DataFrame(encoded_teams, columns=one_hot_encoder.get_feature_names_out())

    # Делаем предсказания
    total_score_pred = model_total.predict(encoded_teams_df)[0]
    home_score_pred = model_home.predict(encoded_teams_df)[0]
    away_score_pred = model_away.predict(encoded_teams_df)[0]
    handicap_pred = model_handicap.predict(encoded_teams_df)[0]

    # Возвращаем результаты
    return jsonify({
        'total_score': total_score_pred,
        'home_score': home_score_pred,
        'away_score': away_score_pred,
        'handicap': handicap_pred
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
