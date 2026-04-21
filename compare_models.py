import os
import time
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Disable annoying TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # GPU Memory Fix
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
OUTPUT_DIR = 'comparison' # Folder to save the final chart

# Create the comparison directory right away if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keras Models (No EfficientNet)
KERAS_MODELS = {
    'MobileNetV2': 'models/mobilenet_model.keras',
    'ResNet50': 'models/resnet_model.keras',
    'VGG16': 'models/vgg_model.keras'
}

# Frozen Graph (.pb) Model Settings
PB_MODEL_NAME = 'InceptionV3'
PB_MODEL_PATH = 'logs/output_graph.pb'
PB_INPUT_TENSOR = 'ExpandDims:0'     
PB_OUTPUT_TENSOR = 'final_result:0'  
PB_IMG_SIZE = (299, 299)             

print("\n--- Loading Validation Data (224x224) ---")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", 
    seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    shuffle=False # Keep it ordered
)

names, accuracies, losses, inference_times = [], [], [], []

# ==========================================
# 1. EVALUATE KERAS MODELS
# ==========================================
for name, model_path in KERAS_MODELS.items():
    if not os.path.exists(model_path):
        print(f"[SKIP] {name} not found at {model_path}")
        continue
        
    print(f"\nEvaluating {name} (Keras)...")
    model = tf.keras.models.load_model(model_path)
    
    loss, accuracy = model.evaluate(val_dataset, verbose=0)
    
    # Warmup
    dummy_data = np.zeros((1, 224, 224, 3))
    model.predict(dummy_data, verbose=0)
    
    # Time speed 
    start_time = time.time()
    for images, labels in val_dataset.take(10):
        model.predict(images, verbose=0)
    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / (10 * BATCH_SIZE)) * 1000
    
    print(f"  Accuracy: {accuracy*100:.2f}% | Loss: {loss:.2f} | Speed: {avg_time_ms:.2f} ms")
    names.append(name); accuracies.append(accuracy * 100); losses.append(loss); inference_times.append(avg_time_ms)
    
    # FREE MEMORY: Clear the loaded model so it doesn't hoard RAM
    del model
    tf.keras.backend.clear_session()
    gc.collect()

# ==========================================
# 2. EVALUATE PB MODEL (INCEPTION V3)
# ==========================================
if os.path.exists(PB_MODEL_PATH):
    print(f"\nEvaluating {PB_MODEL_NAME} (.pb Frozen Graph)...")
    
    # 1. Load the frozen graph into its own sandbox first
    pb_graph = tf.Graph()
    with pb_graph.as_default():
        with tf.io.gfile.GFile(PB_MODEL_PATH, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        
        try:
            input_layer = pb_graph.get_tensor_by_name(PB_INPUT_TENSOR)
            output_layer = pb_graph.get_tensor_by_name(PB_OUTPUT_TENSOR)
            sess = tf.compat.v1.Session(graph=pb_graph)
            
            # Warmup
            dummy_data = np.zeros((1, 299, 299, 3))
            sess.run(output_layer, feed_dict={input_layer: dummy_data})
        except KeyError as e:
            print(f"\n[ERROR] Could not find the tensor name in the .pb file: {e}")
            sess = None

    # 2. Iterate dynamically OUTSIDE the graph context.
    # This prevents the TF1/TF2 clash AND stops the Memory Leak!
    if sess is not None:
        correct_guesses = 0
        total_images = 0
        total_loss = 0
        scce_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

        start_time = time.time()

        # Lazy loading: We only grab one batch at a time
        for batch_images, batch_labels in val_dataset:
            # Resize just this single batch and convert to numpy
            resized_images = tf.image.resize(batch_images, PB_IMG_SIZE).numpy()
            
            batch_predictions = []
            
            # Feed images one by one to the PB model
            for i in range(resized_images.shape[0]):
                single_image = np.expand_dims(resized_images[i], axis=0) 
                preds = sess.run(output_layer, feed_dict={input_layer: single_image})
                batch_predictions.append(preds[0])
                
            predictions = np.array(batch_predictions)
            
            # Metrics
            predicted_classes = np.argmax(predictions, axis=1)
            correct_guesses += np.sum(predicted_classes == batch_labels.numpy())
            total_images += len(batch_labels)
            
            batch_loss = scce_loss_fn(batch_labels, predictions).numpy()
            total_loss += batch_loss * len(batch_labels)

        end_time = time.time()
        sess.close() # Close session to free memory
        
        pb_accuracy = correct_guesses / total_images
        pb_loss = total_loss / total_images
        pb_speed = ((end_time - start_time) / total_images) * 1000

        print(f"  Accuracy: {pb_accuracy*100:.2f}% | Loss: {pb_loss:.2f} | Speed: {pb_speed:.2f} ms")
        
        names.append(PB_MODEL_NAME)
        accuracies.append(pb_accuracy * 100)
        losses.append(pb_loss)
        inference_times.append(pb_speed)
else:
    print(f"\n[SKIP] {PB_MODEL_NAME} not found at {PB_MODEL_PATH}")

# ==========================================
# 3. PLOT RESULTS & SAVE TO FOLDER
# ==========================================
if len(names) > 0:
    print("\n--- Generating Comparison Charts ---")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    colors = colors[:len(names)]

    # Plot 1: Accuracy
    axes[0].bar(names, accuracies, color=colors)
    axes[0].set_title('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(accuracies) - 5 if accuracies else 0, 100])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot 2: Loss
    axes[1].bar(names, losses, color=colors)
    axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Loss Value')
    axes[1].tick_params(axis='x', rotation=45)

    # Plot 3: Speed
    axes[2].bar(names, inference_times, color=colors)
    axes[2].set_title('Inference Speed (ms/image)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Milliseconds (ms)')
    axes[2].tick_params(axis='x', rotation=45)

    # Format text on bars
    for i, v in enumerate(accuracies): axes[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
    for i, v in enumerate(losses): axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    for i, v in enumerate(inference_times): axes[2].text(i, v + 0.5, f"{v:.1f}ms", ha='center', fontweight='bold')

    plt.tight_layout()
    
    # Save chart inside the new folder
    save_path = os.path.join(OUTPUT_DIR, "model_comparison_chart_all.png")
    plt.savefig(save_path, dpi=300)
    print(f"Done! Chart saved successfully as '{save_path}'")