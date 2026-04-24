import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, classification_report)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # GPU Memory Fix for RTX 3050
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

print("\n--- Running Full Model Evaluation (Matrices & Metrics) ---")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = './dataset'
MODELS_DIR = './models'
OUTPUT_DIR = './evaluation_results'

# Ensure base output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Datasets
print("Loading training dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123, 
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

print("Loading validation dataset...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123, 
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = val_dataset.class_names

# Extract true labels
y_true_train = np.concatenate([y for x, y in train_dataset], axis=0)
y_true_val = np.concatenate([y for x, y in val_dataset], axis=0)

# 2. Find all saved models
if not os.path.exists(MODELS_DIR):
    print(f"Error: Could not find the '{MODELS_DIR}' directory.")
    exit()

saved_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras')]

if not saved_models:
    print(f"No .keras models found in '{MODELS_DIR}'.")
    exit()

# List to hold metrics for the final summary table
all_models_metrics = []

# 3. Process each model
for model_file in saved_models:
    model_name = model_file.replace('.keras', '')
    model_path = os.path.join(MODELS_DIR, model_file)
    
    print(f"\n========================================")
    print(f"Evaluating Model: {model_name}")
    print(f"========================================")
    
    # Load Model
    model = tf.keras.models.load_model(model_path)
    
    # Get Predictions
    print("Generating predictions on Training Data...")
    y_pred_train = np.argmax(model.predict(train_dataset), axis=1)
    
    print("Generating predictions on Validation Data...")
    y_pred_val = np.argmax(model.predict(val_dataset), axis=1)
    
    # Calculate Core Metrics
    train_acc = accuracy_score(y_true_train, y_pred_train)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    precision = precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
    recall = recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
    f1 = f1_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
    
    # --- CREATE PER-MODEL TABLE ---
    model_metrics_dict = {
        "Model Name": model_name,
        "Train Acc": f"{train_acc * 100:.2f}%",
        "Val Acc": f"{val_acc * 100:.2f}%",
        "Precision": f"{precision * 100:.2f}%",
        "Recall": f"{recall * 100:.2f}%",
        "F1 Score": f"{f1 * 100:.2f}%"
    }
    all_models_metrics.append(model_metrics_dict)
    
    print("\n--- Model Metrics Table ---")
    single_model_df = pd.DataFrame([model_metrics_dict])
    print(single_model_df.to_string(index=False))
    print("---------------------------\n")

    # Generate Full Classification Report (Breakdown per class)
    class_report = classification_report(y_true_val, y_pred_val, target_names=class_names, zero_division=0)
    
    # Create the specific folder for this model
    model_output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # --- SAVE METRICS TO TEXT FILE ---
    metrics_path = os.path.join(model_output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("----------------------------------------\n")
        f.write(single_model_df.to_string(index=False) + "\n")
        f.write("----------------------------------------\n")
        f.write("Detailed Validation Report per Class:\n\n")
        f.write(class_report)
    
    print(f"Saved metrics text file to: {metrics_path}")
    
    # --- SAVE CONFUSION MATRIX ---
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_val, y_pred_val)
    
    plt.figure(figsize=(16, 14)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix: {model_name}\nVal Acc: {val_acc*100:.2f}% | F1: {f1*100:.2f}%', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    matrix_path = os.path.join(model_output_dir, f'{model_name}.png')
    plt.savefig(matrix_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved matrix to:  {matrix_path}")

# --- CREATE FINAL SUMMARY TABLE FOR ALL MODELS ---
print("\n" + "="*80)
print("FINAL EVALUATION SUMMARY (ALL MODELS)")
print("="*80)

summary_df = pd.DataFrame(all_models_metrics)
print(summary_df.to_string(index=False))

# Save the summary table to a CSV file for easy viewing
summary_csv_path = os.path.join(OUTPUT_DIR, 'summary_all_models.csv')
summary_df.to_csv(summary_csv_path, index=False)

print("="*80)
print(f"Saved summary table as CSV to: {summary_csv_path}")
print("All evaluations complete!")