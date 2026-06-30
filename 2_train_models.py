import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, 
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

# TensorFlow / Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. PATH AND FOLDER CONFIGURATIONS
# ---------------------------------------------------------
run_mode = "raw"  # Change to 'augmented' for data augmentation mode

figures_dir = f"results/figures/{run_mode}"
tables_dir = "results/tables"
dataset_dir = "data/processed_datasets"

# Create directories if they do not exist
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# Load the dataset from processed directory
with open(f'{dataset_dir}/data_{run_mode}.pkl', 'rb') as file:
    data_dict = pickle.load(file)

class_names = ['0 ng', '0.005 ng', '0.010 ng', '0.025 ng', '0.035 ng',
               '0.050 ng', '01 ng', '05 ng', '10 ng', '15 ng', '20 ng', '25 ng']

# ---------------------------------------------------------
# 2. MODELS AND HELPER FUNCTIONS
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=40),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=40),
    "Gradient Boosting": GradientBoostingRegressor(random_state=40),
    "GaussianProcessRegressor": GaussianProcessRegressor(random_state=40),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "MLP Regression": MLPRegressor(random_state=40)
}

def get_predicted_class(y_pred):
    """Rounds regression predictions to the nearest class index"""
    return np.round(y_pred).clip(0, 11).astype(int)

# Font configurations
font_title = {"family": "Arial", "weight": "bold", "size": 20}
font_axis = {"family": "Arial", "weight": "bold", "size": 18}
font_label = {"family": "Arial", "weight": "bold", "size": 18}
font_ticks = {"family": "Arial", "weight": "bold", "size": 18}
font_tick_size = 18

def plot_and_save(name, y_test, y_pred_raw, y_pred_class, metrics, window_start, window_end, run_mode):
    """Saves Confusion Matrix, Scatter Plot, and Box Plot directly to the figures directory."""
    min_val = min(min(y_test), min(y_pred_raw))
    max_val = max(max(y_test), max(y_pred_raw))
    metrics_text = f"R²: {metrics['R2']:.4f}\nMAE: {metrics['MAE']:.4f}\nMSE: {metrics['MSE']:.4f}\nRMSE: {metrics['RMSE']:.4f}\nEVS: {metrics['EVS']:.4f}"

    # 1. Confusion Matrix
    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred_class), index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False, annot_kws={"size": 16, "weight": "bold"})
    plt.xlabel('Predicted Class', fontdict=font_axis)
    plt.ylabel('True Class', fontdict=font_axis)
    plt.xticks(fontsize=font_tick_size, fontweight=font_ticks['weight'], rotation=45)
    plt.yticks(fontsize=font_tick_size, fontweight=font_ticks['weight'])
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/{name}_{window_start}-{window_end}_Window_conf.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

    # 2. Scatter Plot
    plt.figure(figsize=(8, 6), dpi=600)
    plt.scatter(y_test, y_pred_raw, alpha=0.5, label='Test Points')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values', fontdict=font_axis)
    plt.ylabel('Predicted Values', fontdict=font_axis)
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.xticks(fontsize=font_tick_size, fontweight=font_ticks['weight'])
    plt.yticks(fontsize=font_tick_size, fontweight=font_ticks['weight'])
    plt.grid(True, alpha=0.3)
    plt.legend(prop=font_label, loc=4, frameon=True, facecolor='white')
    plt.text(0.05, 0.95, metrics_text, ha='left', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.3, edgecolor='black'), fontdict=font_label)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/{name}_{window_start}-{window_end}_Window_scatter.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

    # 3. Box Plot
    data_figure = pd.DataFrame({'True Class': y_test, 'Predicted Value': y_pred_raw})
    plt.figure(figsize=(8, 6), dpi=600)
    sns.boxplot(x='True Class', y='Predicted Value', data=data_figure, palette='Blues')
    plt.xlabel('True Values', fontdict=font_axis)
    plt.ylabel('Predicted Values', fontdict=font_axis)
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.xticks(fontsize=font_tick_size, fontweight=font_ticks['weight'])
    plt.yticks(fontsize=font_tick_size, fontweight=font_ticks['weight'])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect\nPrediction')
    plt.legend(prop=font_label, loc=4, frameon=True)
    plt.text(0.05, 0.95, metrics_text, ha='left', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.3, edgecolor='black'), fontdict=font_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/{name}_{window_start}-{window_end}_Window_box.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# 3. MAIN LOOP (WINDOWING, ML + 1D-CNN)
# ---------------------------------------------------------
results = []
window_starts = [100, 90, 80, 70, 60]
window_ends = [120, 130, 140, 150, 160]

for i in range(len(window_starts)):
    selected_classes = list(data_dict.keys())
    window_start, window_end = window_starts[i], window_ends[i]

    print(f"\n{'='*50}\n{window_start}-{window_end} WINDOW TRAININGS\n{'='*50}")

    windowed_dict = {cls_key: [series[window_start:window_end] for series in data_dict[cls_key]] for cls_key in selected_classes}
    
    X, y = [], []
    for cls_key in windowed_dict.keys():
        cls_num = int(cls_key.split('_')[1])
        for series in windowed_dict[cls_key]:
            X.append(series)
            y.append(cls_num)

    X, y = np.array(X), np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- A) TRADITIONAL MACHINE LEARNING MODELS ---
    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train_scaled, y_train)
        
        y_pred_raw = model.predict(X_test_scaled)
        y_pred_class = get_predicted_class(y_pred_raw)

        metrics = {
            'R2': r2_score(y_test, y_pred_raw),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_raw)),
            'MAE': mean_absolute_error(y_test, y_pred_raw),
            'MSE': mean_squared_error(y_test, y_pred_raw),
            'EVS': explained_variance_score(y_test, y_pred_raw),
            'ACC': accuracy_score(y_test, y_pred_class),
            'PREC': precision_score(y_test, y_pred_class, average='macro'),
            'REC': recall_score(y_test, y_pred_class, average='macro'),
            'F1': f1_score(y_test, y_pred_class, average='macro')
        }
        
        metrics['Model'] = name
        metrics['Window'] = f"{window_start}-{window_end}"
        results.append(metrics)
        plot_and_save(name, y_test, y_pred_raw, y_pred_class, metrics, window_start, window_end, run_mode)

    # --- B) 1D-CNN MODEL INTEGRATION ---
    print("Training: 1D-CNN (Deep Learning)")
    X_train_cnn = X_train_scaled[..., np.newaxis]
    X_test_cnn = X_test_scaled[..., np.newaxis]

    cnn_model = Sequential([
        Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=64, kernel_size=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(filters=128, kernel_size=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=1),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1) 
    ])

    cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    y_pred_raw_cnn = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    y_pred_class_cnn = get_predicted_class(y_pred_raw_cnn)

    cnn_metrics = {
        'R2': r2_score(y_test, y_pred_raw_cnn),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_raw_cnn)),
        'MAE': mean_absolute_error(y_test, y_pred_raw_cnn),
        'MSE': mean_squared_error(y_test, y_pred_raw_cnn),
        'EVS': explained_variance_score(y_test, y_pred_raw_cnn),
        'ACC': accuracy_score(y_test, y_pred_class_cnn),
        'PREC': precision_score(y_test, y_pred_class_cnn, average='macro'),
        'REC': recall_score(y_test, y_pred_class_cnn, average='macro'),
        'F1': f1_score(y_test, y_pred_class_cnn, average='macro'),
        'Model': '1D-CNN',
        'Window': f"{window_start}-{window_end}"
    }
    
    results.append(cnn_metrics)
    plot_and_save('1D-CNN', y_test, y_pred_raw_cnn, y_pred_class_cnn, cnn_metrics, window_start, window_end, run_mode)

# ---------------------------------------------------------
# 4. EXCEL EXPORTS
# ---------------------------------------------------------
print("\nExporting Excel spreadsheets to 'results/tables' directory...")
df = pd.DataFrame(results)

# 1. Vertical Export
df.to_excel(f"{tables_dir}/results_{run_mode}_vertical.xlsx", index=False)

# 2. Horizontal Export (Models aligned side-by-side)
chunk_size = 9 
chunks = [df.iloc[i:i + chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]

for i in range(1, len(chunks)):
    chunks[i] = chunks[i].drop(columns=["Model", "Window"])

result_horizontal = pd.concat(chunks, axis=1)
result_horizontal.to_excel(f"{tables_dir}/results_{run_mode}_horizontal.xlsx", index=False)

# 3. Top 3 Export
columns_to_drop = ["MSE", "EVS", "ACC", "PREC", "REC", "F1"]
df_cleaned = result_horizontal.drop(columns=columns_to_drop, errors='ignore')
df_cleaned.to_excel(f"{tables_dir}/results_{run_mode}_top3.xlsx", index=False)

print(f"ALL PROCESSES COMPLETED SUCCESSFULLY! Check '{tables_dir}' and '{figures_dir}' for results.")