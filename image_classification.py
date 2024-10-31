from PIL import Image
import cv2
import numpy as np
import torch
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load pre-trained models and processors
gender_classification_processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
gender_classification_model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

hair_color_classification_processor = AutoImageProcessor.from_pretrained("enzostvs/hair-color")
hair_color_classification_model = AutoModelForImageClassification.from_pretrained("enzostvs/hair-color")

model_dir = os.path.abspath("./models/")
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Load pre-trained models for person detection
person_detection_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Define hair color labels
hair_color_labels = ["red hair", "blond hair", "white hair", "black hair", "completely bald"]

# Map class ID to hair color label
def map_class_id_to_hair_color(class_id):
    if 0 <= class_id < len(hair_color_labels):
        return hair_color_labels[class_id]
    else:
        return "Unknown"  # Handle out-of-range class IDs

# Detect if there is a person in the image
def detect_person(image):
    # Prepare the image for the detection model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    person_detection_model.setInput(blob)
    detections = person_detection_model.forward()

    # Check for person detection
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            return True
    return False

# Classify hair color in the image
def classify_hair_color(image):
    # Preprocess the image for the hair_color classification model
    processed_image = hair_color_classification_processor(images=image, return_tensors="pt").pixel_values

    # Predict hair color
    with torch.no_grad():
        hair_color_prediction = hair_color_classification_model(processed_image)

    # Get predicted class ID using argmax
    predicted_class_id = torch.argmax(hair_color_prediction.logits, dim=1).item()

    # Map class ID to hair color label
    hair_color_label = map_class_id_to_hair_color(predicted_class_id)

    print(f"Predicted hair color: {hair_color_label}")
    return hair_color_label

# Classify gender in the image
def classify_gender(image):
    # Preprocess the image for the gender classification model
    processed_image = gender_classification_processor(images=image, return_tensors="pt").pixel_values

    # Predict gender
    with torch.no_grad():
        gender_prediction = gender_classification_model(processed_image)

    gender = gender_classification_model.config.id2label[torch.argmax(gender_prediction.logits, dim=1).item()]
    # print(f"Predicted gender: {gender}")
    return gender

# Main function to process the image and check for gender and hair color
def classify_by_desire(image_path, desired_gender, desired_hair_color=None):
    if isinstance(image_path, str):
        image = cv2.imread(image_path)  # Read the image from a file path
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB for model processing
    elif isinstance(image_path, Image.Image):
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)

    if detect_person(image):
        gender = classify_gender(image)
        print(gender, flush=True)
        if gender == desired_gender:
            hair_color = classify_hair_color(image)
            if hair_color == desired_hair_color or not desired_hair_color:
                print(f"The person in the photo is a {hair_color} {desired_gender}.")
                return { "status": True, "hair_color": hair_color}
            else:
                if hair_color == "completely bald":
                    return { "status": True, "hair_color": hair_color}
                else:
                    print(f"The person in the photo is a {gender}, but has {hair_color}.")
        else:
            print(f"The person in the photo is not a {desired_gender}.")
    else:
        print("No person detected in the photo.")
        return { "status": True }

# Example usage
image_paths = ['test.jpeg', 'julia.jpeg', 'completely_bald.jpg']
desired_gender = 'female'
desired_hair_color = 'black hair'

for image_path in image_paths:
    classify_by_desire(image_path, desired_gender, desired_hair_color)
