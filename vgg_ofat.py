import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

print("\n--- Training VGG16 (Automated OFAT Testing) ---")

ofat_tests = [
#    {"name": "1_Baseline", "batch_size": 32, "lr": 0.001, "dropout": 0.2, "optimizer": 'adam'},
    {"name": "2_BatchUP",  "batch_size": 64, "lr": 0.001, "dropout": 0.2, "optimizer": 'adam'},
    {"name": "3_BatchDN",  "batch_size": 16, "lr": 0.001, "dropout": 0.2, "optimizer": 'adam'},
    {"name": "4_LR_UP",    "batch_size": 32, "lr": 0.01,  "dropout": 0.2, "optimizer": 'adam'},
    {"name": "5_LR_DN",    "batch_size": 32, "lr": 0.0001,"dropout": 0.2, "optimizer": 'adam'},
    {"name": "6_DropoutUP","batch_size": 32, "lr": 0.001, "dropout": 0.5, "optimizer": 'adam'},
    {"name": "7_Optimizer","batch_size": 32, "lr": 0.001, "dropout": 0.2, "optimizer": 'sgd'}
]

IMG_SIZE = (224, 224)
DATASET_DIR = './dataset'
os.makedirs('models', exist_ok=True)

for test in ofat_tests:
    print(f"\n=======================================================")
    print(f"Running Test: {test['name']}")
    print(f"Config: Batch={test['batch_size']}, LR={test['lr']}, Dropout={test['dropout']}, Opt={test['optimizer']}")
    print(f"=======================================================")

    tf.keras.backend.clear_session()

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=123, image_size=IMG_SIZE, batch_size=test['batch_size'])
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=123, image_size=IMG_SIZE, batch_size=test['batch_size'])

    class_names = train_dataset.class_names
    if test['name'] == "1_Baseline":
        with open('modern_labels.txt', 'w') as f:
            f.write('\n'.join(class_names))

    base_model = applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(test['dropout']),
        layers.Dense(len(class_names), activation='softmax')
    ])

    opt = optimizers.Adam(learning_rate=test['lr']) if test['optimizer'] == 'adam' else optimizers.SGD(learning_rate=test['lr'])
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=5, 
        restore_best_weights=True, 
        verbose=1
        )
    
    model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=50, 
        callbacks=[early_stopping]
        )

    save_path = os.path.join('models', f"vgg_b{test['batch_size']}_lr{test['lr']}_do{test['dropout']}_{test['optimizer']}.keras")
    model.save(save_path)
    print(f"\nSaved as '{save_path}'")