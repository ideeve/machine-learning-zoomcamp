import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from itertools import product
import xgboost as xgb
from xgboost import DMatrix
from math import sqrt

data_url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"

def load_and_prepare_data(data_source):
    housing_data = pd.read_csv(data_source)
    selected_data = housing_data[housing_data['region'].isin(['Region A', 'Region B'])]
    cleaned_data = selected_data.fillna(0)
    features = cleaned_data.drop('target_value', axis=1)
    target = np.log1p(cleaned_data['target_value'])

    X_train_all, X_test, y_train_all, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.25, random_state=42)

    dict_vectorizer = DictVectorizer(sparse=False)
    train_dict = X_train.to_dict(orient='records')
    X_train_encoded = dict_vectorizer.fit_transform(train_dict)

    val_dict = X_val.to_dict(orient='records')
    X_val_encoded = dict_vectorizer.transform(val_dict)

    test_dict = X_test.to_dict(orient='records')
    X_test_encoded = dict_vectorizer.transform(test_dict)

    return cleaned_data, X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test

def train_and_evaluate(X_train, X_val, y_train, y_val, data):
    decision_tree = DecisionTreeRegressor(max_depth=1)
    decision_tree.fit(X_train, y_train)
    y_pred_train = decision_tree.predict(X_train)
    y_pred_val = decision_tree.predict(X_val)
    train_mse = mean_squared_error(y_train, y_pred_train)
    val_mse = mean_squared_error(y_val, y_pred_val)
    print('Decision Tree - Train MSE:', train_mse)
    print('Decision Tree - Validation MSE:', val_mse)
    tree_text = export_text(decision_tree, feature_names=data.columns.tolist())
    print('Decision Tree Structure:')
    print(tree_text)

def train_random_forest(X_train, X_val, y_train, y_val):
    random_forest = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    y_pred_val = random_forest.predict(X_val)
    val_rmse = sqrt(mean_squared_error(y_val, y_pred_val))
    print('Random Forest - Validation RMSE:', val_rmse)

def experiment_with_n_estimators(X_train, X_val, y_train, y_val):
    best_n_estimators = None
    best_rmse = float('inf')
    for n_estimators in range(10, 201, 10):
        random_forest = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        random_forest.fit(X_train, y_train)
        y_pred_val = random_forest.predict(X_val)
        rmse_val = sqrt(mean_squared_error(y_val, y_pred_val))
        print(f'n_estimators={n_estimators}, Validation RMSE: {rmse_val:.3f}')
        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_n_estimators = n_estimators
        else:
            break
    print(f"RMSE stopped improving after n_estimators={best_n_estimators}")

def select_best_max_depth(X_train, X_val, y_train, y_val):
    best_max_depth = None
    best_n_estimators = None
    best_mean_rmse = float('inf')
    max_depth_values = [10, 15, 20, 25]
    n_estimators_range = range(10, 201, 10)
    for max_depth, n_estimators in product(max_depth_values, n_estimators_range):
        random_forest = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1)
        random_forest.fit(X_train, y_train)
        y_pred_val = random_forest.predict(X_val)
        rmse_val = sqrt(mean_squared_error(y_val, y_pred_val))
        if rmse_val < best_mean_rmse:
            best_mean_rmse = rmse_val
            best_max_depth = max_depth
            best_n_estimators = n_estimators
    print(f"Best max_depth: {best_max_depth}")
    print(f"Best n_estimators: {best_n_estimators}")
    print(f"Best mean RMSE: {best_mean_rmse:.3f}")

def find_most_important_feature(X_train, y_train, data):
    random_forest = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    feature_importances = random_forest.feature_importances_
    feature_names = data.columns[:-1]
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    most_important_feature = max(feature_importance_dict, key=feature_importance_dict.get)
    print("Most Important Feature:", most_important_feature)

def train_xgboost(X_train, X_val, y_train, y_val):
    eta_values = [0.3, 0.1]
    for eta in eta_values:
        xgb_params = {
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'reg:squarederror',
            'nthread': 8,
            'seed': 42,
            'verbosity': 1,
        }
        xgb_params['eta'] = eta
        dtrain = DMatrix(X_train, label=y_train)
        dval = DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        num_round = 100
        bst = xgb.train(xgb_params, dtrain, num_round, watchlist)
        y_pred_val = bst.predict(dval)
        rmse_val = sqrt(mean_squared_error(y_val, y_pred_val))
        print(f'XGBoost - eta={eta}, Validation RMSE: {rmse_val:.3f}')

data, X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test = load_and_prepare_data(data_url)

train_and_evaluate(X_train_encoded, X_val_encoded, y_train, y_val, data)
train_random_forest(X_train_encoded, X_val_encoded, y_train, y_val)
experiment_with_n_estimators(X_train_encoded, X_val_encoded, y_train, y_val)
select_best_max_depth(X_train_encoded, X_val_encoded, y_train, y_val)
find_most_important_feature(X_train_encoded, y_train, data)
train_xgboost(X_train_encoded, X_val_encoded, y_train, y_val)
