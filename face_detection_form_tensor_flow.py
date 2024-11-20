from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from PIL import Image, ImageDraw
from mtcnn import MTCNN

from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras._tf_keras.keras.preprocessing.image import img_to_array,load_img

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MTCNN face detector
detector = MTCNN()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces_with_features(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Detect faces using MTCNN
    faces = detector.detect_faces(image_np)
    valid_faces = []

    # Initialize ImageDraw object
    draw = ImageDraw.Draw(image)

    for face in faces:
        # Extract face bounding box
        x, y, width, height = face['box']
        x1, y1, x2, y2 = x, y, x + width, y + height

        # Extract keypoints (eyes, nose, mouth)
        keypoints = face['keypoints']
        if keypoints:
            eyes_detected = 'left_eye' in keypoints and 'right_eye' in keypoints
            nose_detected = 'nose' in keypoints
            mouth_detected = 'mouth_left' in keypoints and 'mouth_right' in keypoints

            if eyes_detected and nose_detected and mouth_detected:
                valid_faces.append((x1, y1, x2, y2))

    # Draw rectangles around valid faces
    for (x1, y1, x2, y2) in valid_faces:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    # Save the image with detected faces
    result_filename = 'result_' + os.path.basename(image_path)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    image.save(result_path)
    # try:
    #     image.save(result_path)
    #     print(f"Image saved successfully at {result_path}")
    # except Exception as e:
    #     print(f"Error saving the image: {e}")

    return len(valid_faces), result_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Perform face detection with TensorFlow
        num_faces, result_image_filename = detect_faces_with_features(file_path)

        return render_template('result.html', filename=result_image_filename, num_faces=num_faces)
    else:
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
