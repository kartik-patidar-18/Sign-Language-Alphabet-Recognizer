import sys
import os

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# --- MODIFIED: TensorFlow 2.x Compatibility & GPU VRAM Management ---
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Allow GPU memory growth to prevent crashes on 6GB VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# ---------------------------------------------------------------------

# Ensure an image path was provided via command line
if len(sys.argv) < 2:
    print("Usage: python predict_image.py <path_to_image.jpg>")
    sys.exit(1)

image_path = sys.argv[1]

# Check if the file actually exists before trying to read it
if not tf.io.gfile.exists(image_path):
    print(f"Error: Could not find image at '{image_path}'")
    sys.exit(1)

# Read the image_data (Updated to tf.io.gfile)
image_data = tf.io.gfile.GFile(image_path, 'rb').read()

# Loads label file, strips off carriage return (Updated to tf.io.gfile)
label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("logs/output_labels.txt")]

# Unpersists graph from file (Updated to tf.io.gfile)
with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    print(f"\nResults for: {image_path}")
    print("-" * 30)
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))