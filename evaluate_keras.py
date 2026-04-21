import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = './dataset'

MODELS_TO_EVALUATE = [
    ('MobileNetV2', 'models/mobilenet_model.keras'),
    ('ResNet50', 'models/resnet_model.keras'),
    ('VGG16', 'models/vgg_model.keras')
]

print("\n--- Loading Datasets ---")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", 
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", 
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

print("\nExtracting true labels...")
train_true_labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)
val_true_labels = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)

results = []

for model_name, model_file in MODELS_TO_EVALUATE:
    if not os.path.exists(model_file):
        print(f"\n[SKIP] Could not find {model_file}.")
        continue

    print(f"\n--- Evaluating {model_name} ---")
    model = tf.keras.models.load_model(model_file)
    
    print(f"Predicting on Training Data...")
    train_predicted_labels = np.argmax(model.predict(train_dataset, verbose=0), axis=1)
    
    print(f"Predicting on Validation Data...")
    val_predicted_labels = np.argmax(model.predict(val_dataset, verbose=0), axis=1)

    # Metrics
    train_acc = accuracy_score(train_true_labels, train_predicted_labels)
    val_acc = accuracy_score(val_true_labels, val_predicted_labels)
    val_prec = precision_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)
    val_rec = recall_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)
    val_f1 = f1_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)

    results.append({
        'Model': model_name, 'Train Acc': train_acc, 'Val Acc': val_acc,
        'Precision': val_prec, 'Recall': val_rec, 'F1 Score': val_f1
    })

print("\n" + "="*85)
print(f"{'MODEL':<15} | {'TRAIN ACCURACY':<15} | {'VAL ACCURACY':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
print("-" * 85)
for res in results:
    print(f"{res['Model']:<15} | {res['Train Acc']:<15.4f} | {res['Val Acc']:<15.4f} | {res['Precision']:<10.4f} | {res['Recall']:<10.4f} | {res['F1 Score']:<10.4f}")
print("="*85 + "\n")