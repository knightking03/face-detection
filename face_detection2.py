import cv2
import argparse

# Load the pre-trained Haar Cascade face detector provided by OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the pre-trained deep learning model for face detection
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_single_human_face(image_path):
    # Load the image
    image = cv2.imread(image_path)

   # Get the height and width of the image
    h, w = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)

    # Set the input to the model
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    # Loop over the detections and draw boxes around detected faces
    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            count += 1
  
    print(count)

   

    # Display the output image with detected faces
    cv2.imshow('Faces detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Check if exactly one face is detected
    if count < 1:
        print("Error: Could not detect a face.")
        return
    
    if count > 1:
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
