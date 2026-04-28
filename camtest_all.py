import sys
import os
import argparse
import numpy as np
import cv2
import time
import tensorflow as tf

# Suppress warnings for a clean terminal during the live demo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # ==========================================
# # RTX 3050 GPU Memory Fix
# # ==========================================
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("[INFO] GPU Memory Growth Enabled.")
#     except RuntimeError as e:
#         print(e)

def main():
    parser = argparse.ArgumentParser(description="Run the real-time sign language recognizer.")
    parser.add_argument('--model', type=str, default='mobilenet_b32_lr0.001_do0.2_adam', help='Name of the model folder')
    args = parser.parse_args()

    # Automatically target the new folder structure (no .keras extension)
    model_dir = "models"
    model_folder = os.path.join(model_dir, args.model)

    if not os.path.exists(model_folder):
        print(f"\n[ERROR] Could not find '{model_folder}'.")
        print(f"Make sure the model folder is inside your '{model_dir}' directory!")
        sys.exit(1)

    print(f"\nLoading {args.model}... Booting up the pure TF graph.")
    
    # NEW LOAD LOGIC: Universal SavedModel Format
    model = tf.saved_model.load(model_folder)
    infer = model.signatures["serving_default"]

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
    
    print(f"\n✅ System Ready. Webcam started with {args.model}.")
    print("Press 'Esc' on the webcam window to close the demo.")

    while True:
        ret, img = cap.read()
        if not ret:
            continue
            
        img = cv2.flip(img, 1)
        
        # Bounding box coordinates for the hand
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        # Canny Edge Detection Window
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        cv2.imshow("Model Vision (Canny Edges)", edges)
        
        a = cv2.waitKey(1) 
        
        if i == 4:
            # Preprocessing (224x224 RGB)
            img_resized = cv2.resize(img_cropped, (224, 224))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img_rgb, axis=0) # Keep as numpy first
            
            # NEW PREDICTION LOGIC: Raw Execution Graph
            # 1. Convert image to raw float32 tensor
            img_tensor = tf.constant(img_array, dtype=tf.float32)
            
            # 2. Pass it directly to the mathematical execution graph
            raw_predictions = infer(img_tensor)
            
            # 3. Extract the numpy array from the dictionary
            predictions = list(raw_predictions.values())[0].numpy()
            
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
            
            # Draw the loading timer on screen
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
        display_name = args.model.split('_')[0].upper()
        cv2.putText(img, f"Model: {display_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, '%s' % (current_prediction.upper()), (100,420), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (100,460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("Sign Language Alphabet Recognizer", img)
        
        # Bottom sequence sentence viewer
        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.imshow('Sequence Viewer', img_sequence)
        
        if a == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




# To Run this python camtest_all.py --model inception_b64_lr0.001_do0.2_adam