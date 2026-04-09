import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Disable annoying TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Memory Fix for RTX 3050
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = './dataset'
OUTPUT_DIR = 'confusion_matrices' # <--- Name of the new folder

# Tell the script to look inside the 'models/' folder
MODELS_TO_EVALUATE = [
    ('MobileNetV2', 'models/mobilenet_model.keras'),
    ('EfficientNetB0', 'models/efficientnet_model.keras'),
    ('ResNet50', 'models/resnet_model.keras'),
    ('VGG16', 'models/vgg_model.keras')
]

print("\n--- Loading Validation Data ---")
# CRITICAL: shuffle=False! We need the predictions to line up perfectly with the true labels.
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    shuffle=False  # Do not shuffle!
)

# Load the class names
class_names = val_dataset.class_names

print("Extracting true labels... (this might take a few seconds)")
# Unpack the dataset to get the exact list of true answers
true_labels = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)


def plot_confusion_matrix(model_name, y_true, y_pred, classes):
    """Generates and saves a heatmap image of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # --- Create the new folder safely ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Make the plot large enough to read easily
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=16)
    plt.ylabel('True Label (Actual Gesture)', fontsize=12)
    plt.xlabel('Predicted Label (AI Guess)', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Ensure nothing gets cut off
    plt.tight_layout()
    
    # Save the image INSIDE the new folder
    filename = f"confusion_matrix_{model_name.lower()}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  -> Saved {save_path}")


# --- Main Evaluation Loop ---
for model_name, model_file in MODELS_TO_EVALUATE:
    if not os.path.exists(model_file):
        print(f"\n[SKIP] Could not find {model_file}. Did you train this one yet?")
        continue

    print(f"\n--- Evaluating {model_name} ---")
    
    # 1. Load the trained AI
    model = tf.keras.models.load_model(model_file)
    
    # 2. Make it guess every image in the validation set
    print(f"Making predictions with {model_name}...")
    raw_predictions = model.predict(val_dataset)
    
    # 3. Convert percentages into final guesses (e.g., pick the highest probability)
    predicted_labels = np.argmax(raw_predictions, axis=1)
    
    # 4. Draw and save the Confusion Matrix
    print(f"Generating chart for {model_name}...")
    plot_confusion_matrix(model_name, true_labels, predicted_labels, class_names)

print(f"\nAll done! Check the '{OUTPUT_DIR}' folder for your images.")