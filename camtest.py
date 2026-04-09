import sys
import os
import numpy as np
import cv2
import time  

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use TensorFlow 2.x compatibility bridge
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def predict(image_data):
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

# Loads label file
label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("logs/output_labels.txt")]

print("Loading model... Please wait.")
with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        sys.exit()

    i = 0
    sequence = ''
    
    # --- TIMING VARIABLES ---
    current_prediction = ''
    stable_start_time = time.time()
    required_hold_time = 2.0  
    score = 0.0
    
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
        # ---------------------------------------------------------
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # You can tweak these numbers (100, 200) depending on your room's lighting!
        edges = cv2.Canny(blur, 50, 150)
        
        cv2.imshow("Model Vision (Canny Edges)", edges)
        # ---------------------------------------------------------
        
        image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()
        
        a = cv2.waitKey(1) 
        
        if i == 4:
            res_tmp, score = predict(image_data)
            i = 0
            
            if res_tmp != current_prediction:
                current_prediction = res_tmp
                stable_start_time = time.time()  
                
        i += 1
        
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