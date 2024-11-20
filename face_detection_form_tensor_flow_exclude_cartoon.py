from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from PIL import Image, ImageDraw
from mtcnn import MTCNN
from deepface import DeepFace
from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize MTCNN face detector
detector = MTCNN()

# Load VGG16 model
vgg_model = VGG16(weights='imagenet')

# def is_real_face(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)

#     predictions = vgg_model.predict(img_array)
#     decoded_predictions = decode_predictions(predictions, top=500)[0]

#     for _, label, _ in decoded_predictions:
#         print(label)
#         if any(keyword in label for keyword in ['person', 'human', 'man', 'woman', 'face']):
#             return True
#     return False

def is_cartoon(image_path):
    #Use heuristic methods to detect if an image is a cartoon.
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # Calculate edge density (cartoon images tend to have high edge density)
    edge_density = np.sum(edges > 0) / edges.size

    # Calculate color variance (cartoons tend to have lower color variance)
    color_variance = np.var(image)

    # Heuristic thresholds (tune these based on your dataset)
    if edge_density > 0.1 and color_variance < 500:
        return True
    return False

def face_detect_using_deepface(image_path):
#  if not is_cartoon(image_path):
    try:
        result = DeepFace.analyze(image_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
        print(len(result))
        if result:
            return True
        else:
            return False
    except Exception as e:
        print("Error analyzing face:", e)
    return False
#  else:
#         print("Image is recognized as a cartoon and won't be analyzed.")
#         return False

def detect_faces_with_features_tensor(image_path):
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

        # Check if the image is a real face and not a cartoon
        if not face_detect_using_deepface(file_path):
            return render_template('result.html', filename=filename, num_faces=None)

        # Perform face detection with TensorFlow
        num_faces, result_image_filename = detect_faces_with_features_tensor(file_path)

        return render_template('result.html', filename=result_image_filename, num_faces=num_faces)
    else:
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
