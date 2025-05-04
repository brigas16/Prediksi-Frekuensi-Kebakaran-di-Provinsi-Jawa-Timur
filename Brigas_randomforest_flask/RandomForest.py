import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import h5py

# Load and preprocess the data
fire_data = pd.read_excel("Bojonegoro.xlsx")
fire_data = fire_data.sort_values(by='month')

# Group data by month and take the mean of other features
fire_data_grouped = fire_data.groupby('month').mean().reset_index()

# Features and target
X_grouped = fire_data_grouped[[
    'rainfall', 'wind_speed', 'min_temp', 'humidity', 'max_temp']]
y_grouped = fire_data_grouped['num_fires']

# Scale the numerical features
scaler = MinMaxScaler()
X_grouped_scaled = scaler.fit_transform(X_grouped)

# Initialize arrays to hold predictions and metrics for each fold
all_fold_predictions = []
all_r2_scores = []
all_mse_scores = []

# Cross-validation and hyperparameter grid
k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(X_grouped_scaled), 1):
    X_train, X_test = X_grouped_scaled[train_index], X_grouped_scaled[test_index]
    y_train, y_test = y_grouped.iloc[train_index], y_grouped.iloc[test_index]

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                               cv=kf, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict on the entire dataset
    y_pred = best_model.predict(X_grouped_scaled)

    # Store predictions and calculate metrics
    all_fold_predictions.append(y_pred)
    r2 = r2_score(y_grouped, y_pred)
    mse = mean_squared_error(y_grouped, y_pred)
    all_r2_scores.append(r2)
    all_mse_scores.append(mse)

    print(f"\nFold {fold}:")
    for month, prediction in zip(fire_data_grouped['month'], y_pred):
        print(
            f"Month: {month}, Predicted Number of Fires: {int(round(prediction))}")

    print("R-squared (R2) Score:", r2)
    print("Mean Squared Error (MSE):", mse)

# Convert list of predictions to DataFrame for easy manipulation
fold_predictions_df = pd.DataFrame(all_fold_predictions)

# Calculate average prediction for each month across all folds
avg_predictions = fold_predictions_df.mean(axis=0)
avg_r2 = sum(all_r2_scores) / k
avg_mse = sum(all_mse_scores) / k

print("\nAverage Predictions:")
for month, avg_pred in zip(fire_data_grouped['month'], avg_predictions):
    print(
        f"Month: {month}, Average Prediction Number of Fires: {int(round(avg_pred))}")

print("\nAverage R-squared (R2) Score:", avg_r2)
print("Average Mean Squared Error (MSE):", avg_mse)

# Save results to an HDF5 file
with h5py.File('bojonegoro.h5', 'w') as f:
    # Save grouped data
    f.create_dataset('X_grouped', data=X_grouped_scaled)  # Save X_grouped
    f.create_dataset('y_grouped', data=y_grouped.values)
    f.create_dataset('months', data=fire_data_grouped['month'].values)
    f.create_dataset('actual_num_fires', data=y_grouped.values)
    f.create_dataset('avg_predictions', data=avg_predictions.values)

    # Save metrics
    f.create_dataset('all_r2_scores', data=all_r2_scores)
    f.create_dataset('all_mse_scores', data=all_mse_scores)
    f.create_dataset('avg_r2', data=avg_r2)
    f.create_dataset('avg_mse', data=avg_mse)

    # Save feature importances
    f.create_dataset('feature_importances',
                     data=best_model.feature_importances_)
    f.create_dataset('features', data=np.array(X_grouped.columns).astype('S'))

    # Save residuals
    f.create_dataset('residuals', data=(y_grouped - avg_predictions).values)

    # Save monthly predictions and performance metrics
    for i, month in enumerate(fire_data_grouped['month']):
        prediction_dataset_name = f'month_{month}_prediction'
        f.create_dataset(prediction_dataset_name,
                         data=fold_predictions_df.iloc[:, i].values)

        # Use the fold index to access R2 and MSE scores
        r2_score_dataset_name = f'month_{month}_r2_score'
        mse_score_dataset_name = f'month_{month}_mse_score'

        if i < len(all_r2_scores):  # Ensure index is within bounds
            f.create_dataset(r2_score_dataset_name, data=all_r2_scores[i])
            f.create_dataset(mse_score_dataset_name, data=all_mse_scores[i])
        else:
            print(f"Warning: Index {i} out of bounds for R2 and MSE scores")

    # Save the model parameters
    model_group = f.create_group('model')
    model_group.attrs['n_estimators'] = best_model.n_estimators
    model_group.attrs['max_features'] = best_model.max_features
    model_group.attrs['max_depth'] = best_model.max_depth
    model_group.attrs['min_samples_split'] = best_model.min_samples_split
    model_group.attrs['min_samples_leaf'] = best_model.min_samples_leaf

    # Save the scaler parameters
    scaler_group = f.create_group('scaler')
    scaler_group.create_dataset('min_', data=scaler.min_)
    scaler_group.create_dataset('scale_', data=scaler.scale_)
    scaler_group.attrs['n_features_in_'] = scaler.n_features_in_

print("bojonegoro saved to 'bojonegoro1.h5'")
