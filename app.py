from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd
from flask_cors import CORS
import pymongo

from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

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

file_path = 'basketball2.csv'
df = pd.read_csv(file_path)


@app.route('/')
def index():
    # Отображение HTML-шаблона index.html из папки templates
    return render_template('index.html')


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

    # Делаем предсказания и формируем ответ
    predictions = [
        {'Total Score': model_total.predict(encoded_teams_df)[0]},
        {'Home Score': model_home.predict(encoded_teams_df)[0]},
        {'Away Score': model_away.predict(encoded_teams_df)[0]},
        {'1st Quarter Home Score': model_firstQuarterHomeScore.predict(encoded_teams_df)[0]},
        {'1st Quarter Away Score': model_firstQuarterAwayScore.predict(encoded_teams_df)[0]},
        {'2nd Quarter Home Score': model_secondQuarterHomeScore.predict(encoded_teams_df)[0]},
        {'2nd Quarter Away Score': model_secondQuarterAwayScore.predict(encoded_teams_df)[0]},
        {'3rd Quarter Home Score': model_thirdQuarterHomeScore.predict(encoded_teams_df)[0]},
        {'3rd Quarter Away Score': model_thirdQuarterAwayScore.predict(encoded_teams_df)[0]},
        {'4th Quarter Home Score': model_fourthQuarterHomeScore.predict(encoded_teams_df)[0]},
        {'4th Quarter Away Score': model_fourthQuarterAwayScore.predict(encoded_teams_df)[0]}
    ]

    # Возвращаем результаты
    return jsonify(predictions)


@app.route('/team_stats', methods=['POST'])
def team_stats():
    try:
        data = request.json
        home_team = data['homeTeam']
        away_team = data['awayTeam']

        # Фильтрация датафрейма
        filtered_df = df[(df['homeTeam'] == home_team) & (df['awayTeam'] == away_team)]
        
        # Расчет статистик
        total_home_games = len(filtered_df)
        min_total_score = filtered_df['totalScores'].min()
        max_total_score = filtered_df['totalScores'].max()
        min_home_score = filtered_df['home'].min()
        max_home_score = filtered_df['home'].max()
        min_away_score = filtered_df['away'].min()
        max_away_score = filtered_df['away'].max()
        home_wins = (filtered_df['home'] > filtered_df['away']).sum()

        # Преобразование значений в int или float
        stats = [
            {'Total games': int(total_home_games)},
            {'min_total_score': float(min_total_score)},
            {'max_total_score': float(max_total_score)},
            {'min_home_score': float(min_home_score)},
            {'max_home_score': float(max_home_score)},
            {'min_away_score': float(min_away_score)},
            {'max_away_score': float(max_away_score)},
            {'home_wins': int(home_wins)}
        ]

        return jsonify(stats)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred processing your request.'})



@app.route('/input_stats', methods=['POST'])
def input_stats():
    try:
        data = request.json
        home_team = data['homeTeam']
        away_team = data['awayTeam']
        operand = data['operand']
        value = float(data['value'])  # Преобразование строки в число

        # Фильтрация датафрейма
        filtered_df = df[(df['homeTeam'] == home_team) & (df['awayTeam'] == away_team)]
        filtered_df_all_home = df[(df['homeTeam'] == home_team)]
        filtered_df_all_away = df[(df['awayTeam'] == away_team)]        

        # Расчет ожидаемого количества
        total_games = len(filtered_df)
        total_games_home = len(filtered_df_all_home)
        total_games_away = len(filtered_df_all_away)
        
        expected_count = (filtered_df[operand] > value).sum()
        expected_count_home_all = (filtered_df_all_home[operand] > value).sum()
        expected_count_away_all = (filtered_df_all_away[operand] > value).sum()
        
        # Формирование ответа
        stats = [
            {'Total games': int(total_games)},
            {'Total games': int(total_games_home)},
            {'Total games': int(total_games_away)},
            
            {'Operand': operand},
            {'Expected count': int(expected_count)},
            {'Expected count': int(expected_count_home_all)},
            {'Expected count': int(expected_count_away_all)},  
        ]
        return jsonify(stats)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)})





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
