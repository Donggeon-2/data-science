import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Data load
data = pd.read_csv("openpowerlifting.csv")

# Converting Age columns to numeric types
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Remove row with NAN value
data = data.dropna()

# Separation of input attributes and targets
X = data[['BestDeadliftKg', 'BestSquatKg', 'BestBenchKg']]
y = data['TotalKg']

# KFold Configuration
kf = KFold(n_splits=5, shuffle=True, random_state=50)

# Create an empty list for RMSE records
rmse_history = []

# K-Fold cross validation 수행
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Split training data and test data
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and learn random forest models
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    
    # Perform predictions with test data
    y_pred = model.predict(X_test)
    
    # RMSE Calculation and recording
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_history.append(rmse)

# RMSE Records and Average RMSE Output
print("Record RMSE for each split:", rmse_history)
print("Aberage RMSE:", np.mean(rmse_history))


