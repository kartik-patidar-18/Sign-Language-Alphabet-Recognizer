import sys
import os
import argparse
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow import python

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

def main():
    parser = argparse.ArgumentParser(description="Run the real-time sign language recognizer.")
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'resnet', 'vgg'], help='Which AI brain to use')
    args = parser.parse_args()

    model_filename = os.path.join("models", f"{args.model}_model.keras")

    if not os.path.exists(model_filename):
        print(f"Error: Could not find '{model_filename}'. Train it first!")
        sys.exit(1)

    print(f"Loading {args.model.upper()}... This may take a moment.")
    model = tf.keras.models.load_model(model_filename)

    with open("modern_labels.txt", "r") as f:
        label_lines = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        sys.exit()

    i = 0
    sequence = ''
    current_prediction = ''
    stable_start_time = time.time()
    required_hold_time = 2.0  
    score = 0.0
    
    print(f"Webcam started with {args.model.upper()}. Press 'Esc' to close.")

    while True:
        ret, img = cap.read()
        if not ret:
            continue
            
        img = cv2.flip(img, 1)
        
        # Bounding box coordinates
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        # ---------------------------------------------------------
        # --- NEW 3RD WINDOW: Canny Edge Detection ---
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # You can tweak these numbers (100, 200) depending on your room's lighting!
        edges = cv2.Canny(blur, 50, 150)
        
        cv2.imshow("Model Vision (Canny Edges)", edges)
        # ---------------------------------------------------------
        
        a = cv2.waitKey(1) 
        
        if i == 4:
            # Modern Preprocessing (224x224 RGB)
            img_resized = cv2.resize(img_cropped, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = tf.expand_dims(img_rgb, 0)
            
            predictions = model.predict(img_array, verbose=0)
            
            node_id = np.argmax(predictions[0])
            res_tmp = label_lines[node_id]
            new_score = predictions[0][node_id]
            
            i = 0
            
            # Reset timer if prediction changes
            if res_tmp != current_prediction:
                current_prediction = res_tmp
                stable_start_time = time.time() 
            score = new_score
                
        i += 1
        
        # 2-Second Stability Logic
        if current_prediction not in ['nothing', '']:
            elapsed_time = time.time() - stable_start_time
            
            cv2.putText(img, f"Hold stable: {elapsed_time:.1f}s / 2.0s", (100, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if elapsed_time >= required_hold_time:
                if current_prediction == 'space':
                    sequence += ' '
                elif current_prediction == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += current_prediction
                
                stable_start_time = time.time() 
        else:
            stable_start_time = time.time()
            
        # UI Drawing
        cv2.putText(img, f"Model: {args.model.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, '%s' % (current_prediction.upper()), (100,420), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("Sign Language Recognizer", img)
        
        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.imshow('Sequence Viewer', img_sequence)
        
        if a == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# To run the recognizer with different models, use the following commands in your terminal:

# MobileNet: python webcam_modern.py --model mobilenet

# ResNet: python webcam_modern.py --model resnet

# VGG: python webcam_modern.py --model vgg