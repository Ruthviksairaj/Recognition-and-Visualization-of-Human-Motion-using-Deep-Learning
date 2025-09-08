import os
# Set TensorFlow environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load your trained model
model_path = 'C:/Users/ADMIN/Downloads/har video/Video1/my_trained_model.keras'
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['Clapping', 'Meet and Split', 'Sitting', 'Standing Still', 'Walking', 
                'Walking While Reading Book', 'Walking While Using Phone']

def preprocess_video(video_path):
    SEQUENCE_LENGTH = 20
    IMG_SIZE = 112
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while len(frames) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = preprocess_input(frame.astype(np.float32))
        frames.append(frame)
    
    cap.release()
    
    # Pad with zeros if video is too short
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
    
    return np.array(frames)

def predict_action(video_path):
    with tf.device('/GPU:0'):  # Force prediction on GPU
        video = preprocess_video(video_path)
        prediction = model.predict(np.expand_dims(video, axis=0))
        predicted_class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))  # Get confidence score
        predicted_class = class_labels[predicted_class_index]
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    activity_type = None
    confidence = None

    if request.method == 'POST':
        if 'video' not in request.files:
            return 'No file uploaded'
        video_file = request.files['video']
        if video_file.filename == '':
            return 'No selected file'
        
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(video_file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return 'Invalid file type. Please upload a video (MP4/AVI/MOV) or image (JPG/JPEG/PNG)'
        
        if video_file:
            try:
                upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save with appropriate extension
                file_path = os.path.join(upload_dir, f'uploaded_file{file_ext}')
                video_file.save(file_path)
                
                predicted_class, confidence_score = predict_action(file_path)
                prediction = predicted_class
                activity_type = request.form.get('activity_type', '')
                confidence = confidence_score
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            except Exception as e:
                print(f"Error processing file: {str(e)}")
                return "Error processing file"
            
    return render_template('predict.html', 
                         prediction=prediction, 
                         activity_type=activity_type,
                         confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))