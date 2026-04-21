import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
DATASET_DIR = './dataset'
MODEL_NAME = 'InceptionV3'
MODEL_FILE = 'logs/output_graph.pb'
LABELS_FILE = 'logs/output_labels.txt'  # We MUST read this to map predictions!

print("\n--- Resolving Label Mismatches ---")

# 1. Get Keras Alphabetical Labels (This is what our true labels use)
# We use a dummy dataset just to pull the class_names and file paths
dummy_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", 
    seed=123, image_size=(224, 224), batch_size=32, shuffle=False
)
keras_class_names = dummy_dataset.class_names
keras_label_to_index = {name.lower(): idx for idx, name in enumerate(keras_class_names)}

val_true_labels = np.concatenate([y.numpy() for x, y in dummy_dataset], axis=0)
val_paths = dummy_dataset.file_paths

# 2. Get the PB Model's Arbitrary Labels
if not os.path.exists(LABELS_FILE):
    print(f"\n[ERROR] Could not find {LABELS_FILE}. Must have this to map indices.")
    exit()

with open(LABELS_FILE, 'r') as f:
    pb_class_names = [line.strip().lower() for line in f.readlines()]

# 2. Get the PB Model's Arbitrary Labels
if not os.path.exists(LABELS_FILE):
    print(f"\n[ERROR] Could not find {LABELS_FILE}. Must have this to map indices.")
    exit()

with open(LABELS_FILE, 'r') as f:
    pb_class_names = [line.strip().lower() for line in f.readlines()]

# Create a mapping: When PB predicts index 'x', what Keras index is that really?
pb_index_to_keras_index = []
for pb_name in pb_class_names:
    # Handle slight naming variations (e.g., spaces vs underscores)
    clean_name = pb_name.replace(' ', '_')
    
    # --- CRITICAL FIX: EXACT MATCH ONLY ---
    if clean_name in keras_label_to_index:
        pb_index_to_keras_index.append(keras_label_to_index[clean_name])
    else:
        print(f"[WARNING] Unmatched label: PB='{clean_name}'")
        pb_index_to_keras_index.append(0) # Fallback to prevent index crash

print(f"\n--- Evaluating {MODEL_NAME} ---")

with tf.io.gfile.GFile(MODEL_FILE, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name="")
    # CRITICAL FIX: Feed raw JPEG bytes to this specific tensor
    input_tensor = graph.get_tensor_by_name('DecodeJpeg/contents:0')
    output_tensor = graph.get_tensor_by_name('final_result:0')

# Helper function to read raw bytes and map the prediction
def predict_from_paths_raw(paths, sess):
    preds_list = []
    total = len(paths)
    for idx, path in enumerate(paths):
        if idx % 5000 == 0 and idx > 0:
            print(f"  -> Processed {idx}/{total} images...")
            
        # Read the raw JPEG file as bytes (Exactly how retrain.py does it)
        with open(path, 'rb') as f:
            jpeg_data = f.read()
            
        pred_probs = sess.run(output_tensor, {input_tensor: jpeg_data})
        pb_pred_index = np.argmax(pred_probs[0])
        
        # Map the PB index back to the alphabetical Keras index
        keras_mapped_index = pb_index_to_keras_index[pb_pred_index]
        preds_list.append(keras_mapped_index)
        
    return preds_list

with tf.compat.v1.Session(graph=graph) as sess:
    print(f"Predicting on Validation Data ({len(val_paths)} images)...")
    val_predicted_labels = predict_from_paths_raw(val_paths, sess)

print(f"Calculating metrics...")
val_acc = accuracy_score(val_true_labels, val_predicted_labels)
val_prec = precision_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)
val_rec = recall_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)
val_f1 = f1_score(val_true_labels, val_predicted_labels, average='weighted', zero_division=0)

print("\n" + "="*85)
print(f"{'MODEL':<15} | {'VAL ACCURACY':<15} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
print("-" * 85)
print(f"{MODEL_NAME:<15} | {val_acc:<15.4f} | {val_prec:<10.4f} | {val_rec:<10.4f} | {val_f1:<10.4f}")
print("="*85 + "\n")