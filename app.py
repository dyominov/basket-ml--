from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

app = Flask(__name__)

# Загружаем модели и OneHotEncoder
model_total = load('model_total.joblib')
model_home = load('model_home.joblib')
model_away = load('model_away.joblib')
model_firstQuarterAwayScore = load('model_firstQuarterAwayScore.joblib')
model_firstQuarterHomeScore = load('model_firstQuarterHomeScore.joblib')
model_secondQuarterAwayScore = load('model_secondQuarterAwayScore.joblib')
model_secondQuarterHomeScore = load('model_secondQuarterHomeScore.joblib')
model_thirdQuarterAwayScore = load('model_thirdQuarterAwayScore.joblib')
model_thirdQuarterHomeScore = load('model_thirdQuarterHomeScore.joblib')
model_fourthQuarterAwayScore = load('model_fourthQuarterAwayScore.joblib')
model_fourthQuarterHomeScore = load('model_fourthQuarterHomeScore.joblib')
one_hot_encoder = load('one_hot_encoder.joblib')

def get_teams_from_webpage(url):
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tags = soup.find_all('script', {'type': 'application/ld+json'})

    teams = []
    for script_tag in script_tags:
        data = json.loads(script_tag.string)
        if isinstance(data, list):
            for event in data:
                if '@type' in event and event['@type'] == 'SportsEvent':
                    teams.append((event.get('homeTeam', {}).get('name'), event.get('awayTeam', {}).get('name')))
        elif isinstance(data, dict) and '@type' in data and data['@type'] == 'SportsEvent':
            teams.append((data.get('homeTeam', {}).get('name'), data.get('awayTeam', {}).get('name')))
    
    return teams

@app.route('/predict', methods=['GET'])
def predict():
    url = 'https://betwinner-191815.top/ru/live/basketball/2626462-nba-2k24-cyber-league'
    try:
        teams = get_teams_from_webpage(url)
    except Exception as e:
        return jsonify({'error': f'Произошла ошибка при извлечении данных с веб-страницы: {e}'}), 500

    predictions_list = []
    for home_team, away_team in teams:
        # Подготовка данных для прогнозирования
        teams_for_prediction = pd.DataFrame({
            'awayTeam': [away_team],
            'homeTeam': [home_team]
        })
        encoded_teams = one_hot_encoder.transform(teams_for_prediction).toarray()
        encoded_teams_df = pd.DataFrame(encoded_teams, columns=one_hot_encoder.get_feature_names_out())

        # Делаем предсказания и формируем ответ
        predictions = {
            'Match': f"{home_team} vs {away_team}",
            'Total Score': model_total.predict(encoded_teams_df)[0],
            'Home Score': model_home.predict(encoded_teams_df)[0],
            'Away Score': model_away.predict(encoded_teams_df)[0],
            '1st Quarter Home Score': model_firstQuarterHomeScore.predict(encoded_teams_df)[0],
            '1st Quarter Away Score': model_firstQuarterAwayScore.predict(encoded_teams_df)[0],
            '2nd Quarter Home Score': model_secondQuarterHomeScore.predict(encoded_teams_df)[0],
            '2nd Quarter Away Score': model_secondQuarterAwayScore.predict(encoded_teams_df)[0],
            '3rd Quarter Home Score': model_thirdQuarterHomeScore.predict(encoded_teams_df)[0],
            '3rd Quarter Away Score': model_thirdQuarterAwayScore.predict(encoded_teams_df)[0],
            '4th Quarter Home Score': model_fourthQuarterHomeScore.predict(encoded_teams_df)[0],
            '4th Quarter Away Score': model_fourthQuarterAwayScore.predict(encoded_teams_df)[0]
        }
        predictions_list.append(predictions)

    # Возвращаем результаты
    return jsonify(predictions_list)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
