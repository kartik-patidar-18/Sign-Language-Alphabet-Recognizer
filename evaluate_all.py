import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, classification_report)

# 1. Cleaner logging and GPU optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ==========================================
# RTX 3050 GPU Memory Management
# ==========================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU Memory Growth Enabled.")
    except RuntimeError as e:
        print(e)

print("\n--- Running Full Model Evaluation (Full 29x29 Matrix Version) ---")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================
# PATH CONFIGURATION
# ==========================================
DATASET_DIR = 'dataset'
MODELS_DIR = 'models' 
OUTPUT_DIR = 'Model_Evaluations'

os.makedirs(OUTPUT_DIR, exist_ok=True)
summary_csv_path = os.path.join(OUTPUT_DIR, 'summary_all_models.csv')

# Load previous progress if it exists
processed_models = []
if os.path.exists(summary_csv_path):
    existing_df = pd.read_csv(summary_csv_path)
    processed_models = existing_df['Model Name'].tolist()
    print(f"Skipping {len(processed_models)} already evaluated models.")

# 1. Load Datasets
print("\nLoading datasets...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123, 
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123, 
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)

class_names = val_dataset.class_names  # This contains the 29 labels
class_indices = list(range(len(class_names))) # [0, 1, ..., 28]

print("Extracting ground truth labels...")
y_true_train = np.concatenate([y for x, y in train_dataset], axis=0)
y_true_val = np.concatenate([y for x, y in val_dataset], axis=0)

# Helper function for raw graph inference
def get_graph_predictions(dataset, infer_func):
    all_preds = []
    for batch_images, _ in dataset:
        img_tensor = tf.cast(batch_images, tf.float32)
        raw_output = infer_func(img_tensor)
        preds = list(raw_output.values())[0].numpy()
        all_preds.extend(np.argmax(preds, axis=1))
    return np.array(all_preds)

# 2. Iterate through Model Folders
saved_models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]

for model_name in saved_models:
    if model_name in processed_models:
        continue

    model_path = os.path.join(MODELS_DIR, model_name)
    print(f"\n========================================")
    print(f"Processing Model: {model_name}")
    print(f"========================================")
    
    try:
        # Load Universal SavedModel Graph
        model = tf.saved_model.load(model_path)
        infer = model.signatures["serving_default"]
        
        # Inference
        print("Predicting Training Data...")
        y_pred_train = get_graph_predictions(train_dataset, infer)
        
        print("Predicting Validation Data...")
        y_pred_val = get_graph_predictions(val_dataset, infer)
        
        # Calculate Master Metrics
        metrics_dict = {
            "Model Name": model_name,
            "Train Acc": f"{accuracy_score(y_true_train, y_pred_train)*100:.2f}%",
            "Val Acc": f"{accuracy_score(y_true_val, y_pred_val)*100:.2f}%",
            "Precision": f"{precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0)*100:.2f}%",
            "Recall": f"{recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0)*100:.2f}%",
            "F1 Score": f"{f1_score(y_true_val, y_pred_val, average='weighted', zero_division=0)*100:.2f}%"
        }
        
        # Log to console
        print(pd.DataFrame([metrics_dict]).to_string(index=False))

        # Create model output directory
        model_output_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Save classification report (force it to see all 29 labels)
        class_report = classification_report(
            y_true_val, 
            y_pred_val, 
            labels=class_indices, 
            target_names=class_names, 
            zero_division=0
        )
        with open(os.path.join(model_output_dir, 'metrics.txt'), 'w') as f:
            f.write(class_report)
        
        # --- GENERATE FULL 29x29 CONFUSION MATRIX ---
        print("Saving 29x29 Confusion Matrix...")
        cm = confusion_matrix(
            y_true_val, 
            y_pred_val, 
            labels=class_indices
        )
        
        plt.figure(figsize=(20, 18)) # Large size to keep text readable
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix: {model_name} (Acc: {metrics_dict["Val Acc"]})', fontsize=18)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        plt.savefig(os.path.join(model_output_dir, f'{model_name}.png'), bbox_inches='tight')
        plt.close()
        
        # Save progress to CSV
        summary_df = pd.DataFrame([metrics_dict])
        summary_df.to_csv(summary_csv_path, mode='a', header=not os.path.exists(summary_csv_path), index=False)
        
    except Exception as e:
        print(f"FAILED evaluating {model_name}: {e}")
        
    finally:
        # Prevent VRAM buildup
        del model
        gc.collect()
        tf.keras.backend.clear_session()

print("\n" + "="*50)
print("EVALUATION COMPLETE. Check the 'Model_Evaluations' folder.")
print("="*50)