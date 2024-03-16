from flask import Flask, render_template, request, jsonify
from joblib import load
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Загружаем модели и OneHotEncoder
model_total = load('model_total.joblib')
model_home = load('model_home.joblib')
model_away = load('model_away.joblib')
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
        'homeTeam': [home_team],
    })
    encoded_teams = one_hot_encoder.transform(teams_for_prediction).toarray()
    encoded_teams_df = pd.DataFrame(encoded_teams, columns=one_hot_encoder.get_feature_names_out())

    # Делаем предсказания и формируем ответ
    predictions = [
        {'Total Score': model_total.predict(encoded_teams_df)[0]},
        {'Home Score': model_home.predict(encoded_teams_df)[0]},
        {'Away Score': model_away.predict(encoded_teams_df)[0]},
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
        mean_home_score = filtered_df['home'].mean()
        max_home_score = filtered_df['home'].max()
        min_away_score = filtered_df['away'].min()
        mean_away_score = filtered_df['away'].mean()
        max_away_score = filtered_df['away'].max()
        home_wins = (filtered_df['home'] > filtered_df['away']).sum()

        quarter_stats = {}
        quarters = ['firstQuarter', 'secondQuarter', 'thirdQuarter', 'fourthQuarter']
        for quarter in quarters:
            home_score_col = f'{quarter}HomeScore'
            away_score_col = f'{quarter}AwayScore'

            quarter_stats[f'{quarter}_home_mean'] = filtered_df[home_score_col].mean()
            quarter_stats[f'{quarter}_away_mean'] = filtered_df[away_score_col].mean()
            quarter_stats[f'{quarter}_home_max'] = filtered_df[home_score_col].max()
            quarter_stats[f'{quarter}_away_max'] = filtered_df[away_score_col].max()
            quarter_stats[f'{quarter}_home_min'] = filtered_df[home_score_col].min()
            quarter_stats[f'{quarter}_away_min'] = filtered_df[away_score_col].min()

        stats = [
            {'Total games': int(total_home_games)},
            {'min_total_score': float(min_total_score)},
            {'max_total_score': float(max_total_score)},
            {'min_home_score': float(min_home_score)},
            {'mean_home_score': float(mean_home_score)},
            {'max_home_score': float(max_home_score)},
            {'min_away_score': float(min_away_score)},
            {'mean_away_score': float(mean_away_score)},
            {'max_away_score': float(max_away_score)},
            {'home_wins': int(home_wins)}
        ]

        for key, value in quarter_stats.items():
            stats.append({key: float(value)})

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
            {'Operand': operand},
            {'Total games current match': int(total_games)},
            {'Total games for home': int(total_games_home)},
            {'Total games for away': int(total_games_away)},
            {'Expected count current match': int(expected_count)},
            {'Expected count for home': int(expected_count_home_all)},
            {'Expected count for away': int(expected_count_away_all)},
        ]
        return jsonify(stats)
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
