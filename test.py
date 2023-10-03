from nw_ridge_regression import *
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge


def preprocess_nyc_data(file_path):
    df = pd.read_csv(file_path)
    features = ['High Temp (°F)', 'Low Temp (°F)', 'Brooklyn Bridge', 'Manhattan Bridge', 
                'Williamsburg Bridge', 'Queensboro Bridge', 'Precipitation']
    target = 'Total'
    df = df.dropna(subset=features + [target])
    df['Precipitation'] = pd.to_numeric(df['Precipitation'], errors='coerce')
    df['Precipitation'].fillna(0, inplace=True)
    return df[features].values, df[target].values

def preprocess_california_data():
    california = fetch_california_housing()
    return california.data, california.target

def scale_features_target(X, y):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, elapsed_time

def plot_metrics(metrics_df, dataset_name):
    colors = ['red', 'green', 'blue']
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    for i, metric in enumerate(['Time (s)', 'MSE', 'R2']):
        axs[i].barh(metrics_df['Model'], metrics_df[metric], color=colors)
        axs[i].set_xlabel(metric)
        axs[i].set_ylabel('Model')
        axs[i].set_title(f"{metric} ({dataset_name})")
        axs[i].grid(axis='x')
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess datasets
    X_nyc, y_nyc = preprocess_nyc_data('nyc-east-river-bicycle-counts.csv')
    X_nyc_scaled, y_nyc_scaled = scale_features_target(X_nyc, y_nyc)

    X_california, y_california = preprocess_california_data()
    X_california_scaled, y_california_scaled = scale_features_target(X_california, y_california)

    # Initialize models
    svr_model = SVR(kernel='rbf')
    krr_model = KernelRidge(kernel='rbf')
    nwr_model = NadarayaWatsonRidgeRegression(alpha=1.0, h=0.3)

    # Train and evaluate models for each dataset
    datasets = {
        'NYC': (X_nyc_scaled, y_nyc_scaled),
        #'California': (X_california_scaled, y_california_scaled)
    }
    
    for dataset_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics = {'Model': [], 'Time (s)': [], 'MSE': [], 'R2': []}
        
        for model_name, model in zip(['SVR', 'KRR', 'NWR'], [svr_model, krr_model, nwr_model]):
            mse, r2, elapsed_time = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            metrics['Model'].append(model_name)
            metrics['Time (s)'].append(elapsed_time)
            metrics['MSE'].append(mse)
            metrics['R2'].append(r2)
        
        metrics_df = pd.DataFrame(metrics)
        plot_metrics(metrics_df, dataset_name)

main()