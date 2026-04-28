https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train
Sign Language Alphabet Recognizer
A machine learning project designed to train, evaluate, and deploy a model for recognizing sign language alphabets in real-time.

Setup and Installation
1. Download the Dataset

Download the required dataset from Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train

Extract the dataset into the main project directory.

2. Install Dependencies
Ensure your environment is active, then install the necessary packages.

pip install -r requirements.txt

Usage Pipeline
1. Train the Model
Execute the training script one by one for all deep learning models. This will train the network on the dataset and automatically save the resulting model.

python train_inception.py

python train_mobilenet.py

python train_resnet.py

python train_vgg.py

2. Standardize the Model (Optional)
If the resulting model is not standardized, run the export script to format it properly before evaluation or deployment.

python export.py

3. Evaluate the Model
Run the evaluation script to calculate the training accuracy, validation accuracy, precision, recall, and F1 score. This will also generate and save a comprehensive, unified 29x29 confusion matrix for all classes in .png format.

python evaluate.py

4. Run the Live Demo
To run real-time inference using your camera, execute the camtest script while specifying your target model using the --model argument.

python camtest_all.py

python camtest_all.py --model [model name]