import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Memory Fix for RTX 3050
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print("\n--- Training VGG16 (Classic Baseline) ---")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = './dataset'

# Load Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123, image_size=IMG_SIZE, batch_size=BATCH_SIZE)

class_names = train_dataset.class_names
with open('modern_labels.txt', 'w') as f:
    f.write('\n'.join(class_names))

# Build VGG16 Base
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Freeze base model

# Add Custom Classifier Layer
# Note: VGG requires flattening before Dense layers traditionally, but GlobalAveragePooling is more modern and efficient
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- EARLY STOPPING ADDED HERE ---
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',       # Stop based on validation accuracy
    patience=5,                   # Stop if no improvement for 5 epochs
    restore_best_weights=True,    # Automatically keep the best weights
    verbose=1                     # Print a message when early stopping triggers
)

# Train and Save
# (Increased epochs to 50 so Early Stopping has a chance to work)
model.fit(
    train_dataset, 
    validation_data=val_dataset, 
    epochs=50, 
    callbacks=[early_stopping]    # Pass the callback here
)

model.save("vgg_model.keras")
print("\nSaved as 'vgg_model.keras'")

# --- NEW: Create a 'models' folder if it doesn't exist ---
os.makedirs('models', exist_ok=True)

# Save the finalized model inside the models folder
save_path = os.path.join('models', 'vgg_model.keras')
model.save(save_path)
print(f"\nSaved as '{save_path}'")