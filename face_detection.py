import cv2
import argparse

# Load the pre-trained Haar Cascade face detector provided by OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_single_human_face(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale (required for detection)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Filter out false positives using eye detection
    valid_faces = []
    for (x, y, w, h) in faces:
        face_roi = gray_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        if len(eyes) >= 2:  # A valid face should have at least two eyes
            valid_faces.append((x, y, w, h))


    print(len(valid_faces))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in valid_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the output image with detected faces
    cv2.imshow('Faces detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check if exactly one face is detected
    if len(faces) < 1:
        print("Error: Could not detect a face.")
        return
    
    if len(faces) > 1:
        print("Error: Multiple faces detected.")
        return
    


# Create the parser
parser = argparse.ArgumentParser(description="enter name of image")

# Add arguments
parser.add_argument('name', type=str, help="Image Name")

# Parse the arguments
args = parser.parse_args()


# Example usage:
# if __name__ == "__main__":
image_path = args.name  # Replace with your image path
detect_single_human_face(image_path)
