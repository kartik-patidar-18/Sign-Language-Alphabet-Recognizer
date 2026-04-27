import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"\n--- Finishing the Universal Exporter ---")
print(f"Using TensorFlow Version: {tf.__version__}")

INPUT_DIR = './models'
OUTPUT_DIR = './exhibition_models'

if not os.path.exists(INPUT_DIR):
    print(f"Error: Could not find '{INPUT_DIR}'.")
    exit()

saved_models = [f for f in os.listdir(INPUT_DIR) if f.endswith('.keras')]
success_count = 0
skipped_count = 0

for model_file in saved_models:
    input_path = os.path.join(INPUT_DIR, model_file)
    folder_name = model_file.replace('.keras', '')
    output_path = os.path.join(OUTPUT_DIR, folder_name)

    # 1. Skip the Colab models we already successfully converted!
    if os.path.exists(output_path):
        print(f"⏩ {model_file} already exported. Skipping.")
        skipped_count += 1
        continue

    print(f"Exporting Local Model: {model_file}...")

    try:
        # 2. Because you are running this in TF 2.10, it natively reads the local HDF5 format
        model = tf.keras.models.load_model(input_path, compile=False)
        
        # 3. Export as Universal SavedModel Graph
        tf.saved_model.save(model, output_path)
        
        # 4. Erase Keras metadata for a clean load in the webcam script
        metadata_path = os.path.join(output_path, 'keras_metadata.pb')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        print(f"  ✅ Converted to Universal TF Graph!")
        success_count += 1
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

print("\n" + "="*50)
total_ready = skipped_count + success_count
print(f"Final Tally: {skipped_count} Colab models + {success_count} Local models = {total_ready}/{len(saved_models)} ready.")
print("="*50)