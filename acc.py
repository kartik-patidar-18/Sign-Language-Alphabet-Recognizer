import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Disable annoying TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # GPU Memory Fix for RTX 3050 (Uncomment if needed)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = './dataset'

# The list of AI brains we want to test
MODELS_TO_EVALUATE = [
    ('MobileNetV2', 'models/mobilenet_model.keras'),
    ('ResNet50', 'models/resnet_model.keras'),
    ('VGG16', 'models/vgg_model.keras'),
    ('InceptionV3', 'logs/output_graph.pb')
]

# --- Helper Function for TF1 to TF2 Bridge ---
def wrap_frozen_graph(graph_def, inputs, outputs):
    """Wraps a TF1 frozen graph into a modern TF2 callable function."""
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# --- Data Loading ---
print("\n--- Loading Datasets ---")

# Load Training Data (Used for Train Accuracy)
print("Loading Training Data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, 
    validation_split=0.2, 
    subset="training", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    shuffle=False  # Do not shuffle! We need predictions to line up with labels.
)

# Load Validation Data (Used for Val Accuracy, Precision, Recall, F1)
print("Loading Validation Data...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=IMG_SIZE, 
    batch_size=BATCH_SIZE,
    shuffle=False  # Do not shuffle!
)

# Extract true labels
print("\nExtracting true labels... (this might take a few seconds)")
train_true_labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0)
val_true_labels = np.concatenate([y.numpy() for x, y in val_dataset], axis=0)

# List to store results for the final table
results = []

# --- Main Evaluation Loop ---
for model_name, model_file in MODELS_TO_EVALUATE:
    if not os.path.exists(model_file):
        print(f"\n[SKIP] Could not find {model_file}. Did you train this one yet?")
        continue

    print(f"\n--- Evaluating {model_name} ---")
    
    train_preds = []
    val_preds = []

    # Branch logic: Handle .pb (Frozen Graph) differently than .keras
    if model_file.endswith('.pb'):
        print(f"Loading frozen graph (.pb) for {model_name}...")
        
        with tf.io.gfile.GFile(model_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            
        # Wrap the TF1 Graph into a modern TF2 function so we don't break Eager Mode!
        try:
            frozen_func = wrap_frozen_graph(
                graph_def=graph_def,
                inputs=['ExpandDims:0'],
                outputs=['final_result:0']
            )
        except Exception as e:
            print(f"  -> ERROR: Failed to wrap graph: {e}")
            continue

        # Now we can safely iterate through the datasets exactly like normal
        print(f"Predicting on Training Data (Streaming batches to save RAM)...")
        for batch_images, _ in train_dataset:
            # We still predict one by one because of the InceptionV3 graph limitations
            for i in range(batch_images.shape[0]):
                single_image = tf.expand_dims(batch_images[i], axis=0) 
                # Run the wrapped function
                preds = frozen_func(single_image)[0].numpy()
                train_preds.extend(preds)
        
        print(f"Predicting on Validation Data (Streaming batches to save RAM)...")
        for batch_images, _ in val_dataset:
            for i in range(batch_images.shape[0]):
                single_image = tf.expand_dims(batch_images[i], axis=0) 
                preds = frozen_func(single_image)[0].numpy()
                val_preds.extend(preds)
                    
        train_predicted_labels = np.argmax(np.array(train_preds), axis=1)
        val_predicted_labels = np.argmax(np.array(val_preds), axis=1)

    else:
        # Standard TF2 .keras model loading
        print(f"Loading Keras model for {model_name}...")
        model = tf.keras.models.load_model(model_file)
        
        print(f"Predicting on Training Data...")
        raw_train_preds = model.predict(train_dataset, verbose=0)
        train_predicted_labels = np.argmax(raw_train_preds, axis=1)
        
        print(f"Predicting on Validation Data...")
        raw_val_preds = model.predict(val_dataset, verbose=0)
        val_predicted_labels = np.argmax(raw_val_preds, axis=1)

    # --- Calculate Metrics ---
    # Using 'weighted' average for Precision, Recall, and F1 to account for any class imbalance
    print(f"Calculating metrics...")
    
    train_acc = accuracy_score(train_true_labels, train_predicted_labels)
    val_acc = accuracy_score(val_true_labels, val_predicted_labels)
    val_prec = precision_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)
    val_rec = recall_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)
    val_f1 = f1_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)

    # Store results for the table
    results.append({
        'Model': model_name,
        'Train Acc': train_acc,
        'Val Acc': val_acc,
        'Precision': val_prec,
        'Recall': val_rec,
        'F1 Score': val_f1
    })

# --- Final Output Table ---
print("\n" + "="*85)
print(f"{'MODEL':<15} | {'TRAIN ACCURACY':<15} | {'VAL ACCURACY':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
print("-" * 85)
for res in results:
    print(f"{res['Model']:<15} | {res['Train Acc']:<15.4f} | {res['Val Acc']:<15.4f} | {res['Precision']:<10.4f} | {res['Recall']:<10.4f} | {res['F1 Score']:<10.4f}")
print("="*85 + "\n")