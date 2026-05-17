import os

# --- Directory Paths ---
RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
DATA_DIR = os.getenv('DATASET_DIR', 'data/raw')
MODEL_DIR = os.getenv('MODEL_SAVE_DIR', 'saved_models')

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Output File Paths ---
BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, 'baseline_metrics.csv')
PCGRAD_METRICS_PATH = os.path.join(RESULTS_DIR, 'pcgrad_metrics.csv')
GRADIENT_CONFLICT_PATH = os.path.join(RESULTS_DIR, 'gradient_conflict.csv')
FINAL_METRICS_PATH = os.path.join(RESULTS_DIR, 'final_metrics.json')
MODEL_SUMMARY_PATH = os.path.join(RESULTS_DIR, 'model_architecture.txt')

# --- Training Hyperparameters ---
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
EPOCHS = int(os.getenv('EPOCHS', 10))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))