from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import mysql.connector
import h5py
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisissecret'

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login'

# # Function to connect to the database


# def connect_to_database():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="",
#         database="forestfire"
#     )

# # Function to get data from a table


# def get_data_from_table(table_name):
#     db_connection = connect_to_database()
#     cursor = db_connection.cursor()
#     cursor.execute(f"SELECT * FROM {table_name}")
#     data = cursor.fetchall()
#     db_connection.close()
#     return data

# # User class for Flask-Login


# class User(UserMixin):
#     def __init__(self, id, username, password):
#         self.id = id
#         self.username = username
#         self.password = password

# # Loader function for Flask-Login


# @login_manager.user_loader
# def load_user(user_id):
#     db_connection = connect_to_database()
#     cursor = db_connection.cursor()
#     cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
#     user = cursor.fetchone()
#     db_connection.close()
#     if user:
#         return User(user[0], user[1], user[2])
#     return None

# # Route for login page


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         db_connection = connect_to_database()
#         cursor = db_connection.cursor()
#         cursor.execute(
#             "SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
#         user = cursor.fetchone()
#         db_connection.close()
#         if user:
#             login_user(User(user[0], user[1], user[2]))
#             return redirect(url_for('index'))
#         else:
#             flash('Invalid username or password',
#                   'error')  # Flash error message
#     return render_template('login.html')

# Route for the dashboard


@app.route('/')
# @login_required
def index():
    return render_template('index.html')

# Route for data page


@app.route('/data')
# @login_required
def data():
    return render_template('data.html')

# Function to load data from HDF5 file bojonegoro


def load_data():
    with h5py.File('bojonegoro.h5', 'r') as f:
        months = f['months'][:]
        actual_num_fires = f['actual_num_fires'][:]
        avg_predictions = f['avg_predictions'][:]
        avg_r2 = f['avg_r2'][()]
        avg_mse = f['avg_mse'][()]
        feature_importances = f['feature_importances'][:]
        features = f['features'][:]
        residuals = f['residuals'][:]
    return {
        'months': months.tolist(),
        'actual_num_fires': actual_num_fires.tolist(),
        'avg_predictions': avg_predictions.tolist(),
        'avg_r2': float(avg_r2),  # Ensure it's a float
        'avg_mse': float(avg_mse),  # Ensure it's a float
        'feature_importances': feature_importances.tolist(),
        'features': [x.decode() for x in features],
        'residuals': residuals.tolist()
    }


@app.route('/bojonegoro')
def bojonegoro():
    data = load_data()
    return jsonify(data)


@app.route('/visual')
def visual():
    return render_template('visual.html')

# Function to load data from HDF5 file bojonegoro


def load_data1():
    with h5py.File('KabMadiun.h5', 'r') as f:
        months = f['months'][:]
        actual_num_fires = f['actual_num_fires'][:]
        avg_predictions = f['avg_predictions'][:]
        avg_r2 = f['avg_r2'][()]
        avg_mse = f['avg_mse'][()]
        feature_importances = f['feature_importances'][:]
        features = f['features'][:]
        residuals = f['residuals'][:]
    return {
        'months': months.tolist(),
        'actual_num_fires': actual_num_fires.tolist(),
        'avg_predictions': avg_predictions.tolist(),
        'avg_r2': float(avg_r2),  # Ensure it's a float
        'avg_mse': float(avg_mse),  # Ensure it's a float
        'feature_importances': feature_importances.tolist(),
        'features': [x.decode() for x in features],
        'residuals': residuals.tolist()
    }


@app.route('/KabMadiun')
def KabMadiun():
    data1 = load_data1()
    return jsonify(data1)


@app.route('/visual1')
def visual1():
    return render_template('visual1.html')

# Function to load data from HDF5 file bojonegoro


def load_data2():
    with h5py.File('nganjuk.h5', 'r') as f:
        months = f['months'][:]
        actual_num_fires = f['actual_num_fires'][:]
        avg_predictions = f['avg_predictions'][:]
        avg_r2 = f['avg_r2'][()]
        avg_mse = f['avg_mse'][()]
        feature_importances = f['feature_importances'][:]
        features = f['features'][:]
        residuals = f['residuals'][:]
    return {
        'months': months.tolist(),
        'actual_num_fires': actual_num_fires.tolist(),
        'avg_predictions': avg_predictions.tolist(),
        'avg_r2': float(avg_r2),  # Ensure it's a float
        'avg_mse': float(avg_mse),  # Ensure it's a float
        'feature_importances': feature_importances.tolist(),
        'features': [x.decode() for x in features],
        'residuals': residuals.tolist()
    }


@app.route('/nganjuk')
def nganjuk():
    data2 = load_data2()
    return jsonify(data2)


@app.route('/visual2')
def visual2():
    return render_template('visual2.html')

# Function to load data from HDF5 file bojonegoro


def load_data3():
    with h5py.File('ponorogo.h5', 'r') as f:
        months = f['months'][:]
        actual_num_fires = f['actual_num_fires'][:]
        avg_predictions = f['avg_predictions'][:]
        avg_r2 = f['avg_r2'][()]
        avg_mse = f['avg_mse'][()]
        feature_importances = f['feature_importances'][:]
        features = f['features'][:]
        residuals = f['residuals'][:]
    return {
        'months': months.tolist(),
        'actual_num_fires': actual_num_fires.tolist(),
        'avg_predictions': avg_predictions.tolist(),
        'avg_r2': float(avg_r2),  # Ensure it's a float
        'avg_mse': float(avg_mse),  # Ensure it's a float
        'feature_importances': feature_importances.tolist(),
        'features': [x.decode() for x in features],
        'residuals': residuals.tolist()
    }


@app.route('/ponorogo')
def ponorogo():
    data3 = load_data3()
    return jsonify(data3)


@app.route('/visual3')
def visual3():
    return render_template('visual3.html')

# Function to load data from HDF5 file bojonegoro


def load_data4():
    with h5py.File('Situbondo.h5', 'r') as f:
        months = f['months'][:]
        actual_num_fires = f['actual_num_fires'][:]
        avg_predictions = f['avg_predictions'][:]
        avg_r2 = f['avg_r2'][()]
        avg_mse = f['avg_mse'][()]
        feature_importances = f['feature_importances'][:]
        features = f['features'][:]
        residuals = f['residuals'][:]
    return {
        'months': months.tolist(),
        'actual_num_fires': actual_num_fires.tolist(),
        'avg_predictions': avg_predictions.tolist(),
        'avg_r2': float(avg_r2),  # Ensure it's a float
        'avg_mse': float(avg_mse),  # Ensure it's a float
        'feature_importances': feature_importances.tolist(),
        'features': [x.decode() for x in features],
        'residuals': residuals.tolist()
    }


@app.route('/Situbondo')
def Situbondo():
    data4 = load_data4()
    return jsonify(data4)


@app.route('/visual4')
def visual4():
    return render_template('visual4.html')
# Function to load data from HDF5 file


@app.route('/data/ponorogo')
# @login_required
def data_ponorogo():
    data = get_data_from_table('ponorogo')
    return render_template('data.html', data=data)


@app.route('/data/situbondo')
# @login_required
def data_situbondo():
    data = get_data_from_table('situbondo')
    return render_template('data.html', data=data)


@app.route('/data/nganjuk')
# @login_required
def data_nganjuk():
    data = get_data_from_table('nganjuk')
    return render_template('data.html', data=data)


@app.route('/data/bojonegoro')
# @login_required
def data_bojonegoro():
    data = get_data_from_table('bojonegoro')
    return render_template('data.html', data=data)


@app.route('/data/kabmadiun')
# @login_required
def data_kabmadiun():
    data = get_data_from_table('kabmadiun')
    return render_template('data.html', data=data)

# Logout route


# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     flash('You have been logged out', 'info')  # Flash info message
#     return redirect(url_for('login'))


MIN_RAINFALL = 0
MIN_WIND_SPEED = 0
MIN_MIN_TEMP = 17  # Assuming extreme low temperature
MIN_HUMIDITY = 60
MIN_MAX_TEMP = 2

CITIES = ['bojonegoro', 'KabMadiun', 'nganjuk', 'Situbondo', 'ponorogo']

models = {}

for city in CITIES:
    with h5py.File(f'{city}.h5', 'r') as f:
        model_params = {
            'n_estimators': int(f['model'].attrs['n_estimators']),
            'max_features': str(f['model'].attrs['max_features']),
            'max_depth': int(f['model'].attrs['max_depth']),
            'min_samples_split': int(f['model'].attrs['min_samples_split']),
            'min_samples_leaf': int(f['model'].attrs['min_samples_leaf']),
            'random_state': 42  # Ensure the same random state
        }
        scaler_min_ = np.array(f['scaler/min_'][:])
        scaler_scale_ = np.array(f['scaler/scale_'][:])
        X_grouped_scaled = f['X_grouped'][:]
        y_grouped = f['y_grouped'][:]

        scaler = MinMaxScaler()
        scaler.min_ = scaler_min_
        scaler.scale_ = scaler_scale_

        model = RandomForestRegressor(**model_params)
        model.fit(X_grouped_scaled, y_grouped)

        models[city] = {
            'model': model,
            'scaler': scaler,
            'X_grouped_scaled': X_grouped_scaled,
            'y_grouped': y_grouped
        }


@app.route('/predict1')
def predict1():
    return render_template('predict1.html', cities=CITIES)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        city = request.form['city']
        if city not in models:
            raise ValueError('Invalid city selected.')

        month = int(request.form['month'])
        rainfall = float(request.form['rainfall'])
        wind_speed = float(request.form['wind_speed'])
        min_temp = float(request.form['min_temp'])
        humidity = float(request.form['humidity'])
        max_temp = float(request.form['max_temp'])

        # Validate input values
        if rainfall < MIN_RAINFALL:
            raise ValueError(f'Rainfall must be >= {MIN_RAINFALL}.')
        if wind_speed < MIN_WIND_SPEED:
            raise ValueError(f'Wind speed must be >= {MIN_WIND_SPEED}.')
        if min_temp < MIN_MIN_TEMP:
            raise ValueError(f'Minimum temperature must be >= {MIN_MIN_TEMP}.')
        if humidity < MIN_HUMIDITY:
            raise ValueError(f'Humidity must be >= {MIN_HUMIDITY}.')
        if max_temp < MIN_MAX_TEMP:
            raise ValueError(f'Maximum temperature must be >= {MIN_MAX_TEMP}.')

        city_model = models[city]
        model = city_model['model']
        scaler = city_model['scaler']
        X_grouped_scaled = city_model['X_grouped_scaled']
        y_grouped = city_model['y_grouped']

        # Preprocess the input data
        input_data = np.array(
            [[rainfall, wind_speed, min_temp, humidity, max_temp]]
        )
        input_data_scaled = scaler.transform(input_data)

        # Make the prediction
        prediction = model.predict(input_data_scaled)
        prediction_rounded = int(round(prediction[0]))

        # Calculate MSE and R² using the grouped data
        y_pred = model.predict(X_grouped_scaled)
        mse = mean_squared_error(y_grouped, y_pred)
        r2 = r2_score(y_grouped, y_pred)

        return render_template('predict1.html', month=month, prediction=prediction_rounded, mse=mse, r2=r2, cities=CITIES)

    except ValueError as e:
        flash(f'Invalid input: {e}', 'error')
        return redirect(url_for('predict1'))
    except Exception as e:
        flash(f'An error occurred: {e}', 'error')
        return redirect(url_for('predict1'))


@app.route('/metrics', methods=['GET'])
def metrics():
    data = {
        'avg_r2': avg_r2,
        'avg_mse': avg_mse,
        'months': months.tolist(),
        'avg_predictions': avg_predictions.tolist(),
        'actual_num_fires': actual_num_fires.tolist()
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
