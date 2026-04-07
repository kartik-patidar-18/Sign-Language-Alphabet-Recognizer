import sys
import os
import numpy as np
import cv2

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use TensorFlow 2.x compatibility bridge
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def predict(image_data):
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
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

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("logs/output_labels.txt")]

# Unpersists graph from file
print("Loading model... Please wait.")
with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    c = 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        sys.exit()

    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    
    while True:
        ret, img = cap.read()
        if not ret:
            continue
            
        img = cv2.flip(img, 1)
        
        # Bounding box coordinates
        x1, y1, x2, y2 = 100, 100, 300, 300
        img_cropped = img[y1:y2, x1:x2]

        c += 1
        
        # MODIFIED: .tostring() is deprecated. Use .tobytes()
        image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()
        
        a = cv2.waitKey(1) # waits to see if `esc` is pressed
        
        if i == 4:
            res_tmp, score = predict(image_data)
            res = res_tmp
            i = 0
            if mem == res:
                consecutive += 1
            else:
                consecutive = 0
            
            # Sequence building logic (requires 'nothing', 'space', and 'del' classes)
            if consecutive == 2 and res not in ['nothing', '']:
                if res == 'space':
                    sequence += ' '
                elif res == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += res
                consecutive = 0
        i += 1
        
        # Draw main UI elements
        cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
        cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        mem = res
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.imshow("Sign Language Recognizer", img)
        
        # Draw the separate sequence window (Adjusted for better visibility)
        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        cv2.imshow('Sequence Viewer', img_sequence)
        
        if a == 27: # when `esc` is pressed
            break

# MODIFIED: Correctly release the existing cap object
cap.release()
cv2.destroyAllWindows()