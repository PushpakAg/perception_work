import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor
import torch
def compensate_RB(image, flag):
    # Convert to array and split into R, G, B channels
    image_array = np.array(image, dtype=np.float64)
    imageR, imageG, imageB = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

    # Calculate max and min for each channel
    minR, maxR = np.min(imageR), np.max(imageR)
    minG, maxG = np.min(imageG), np.max(imageG)
    minB, maxB = np.min(imageB), np.max(imageB)

    # Normalize pixel values to range (0, 1)
    imageR = (imageR - minR) / (maxR - minR)
    imageG = (imageG - minG) / (maxG - minG)
    imageB = (imageB - minB) / (maxB - minB)

    # Calculate means for each channel
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    # Compensate channels based on flag
    if flag == 0:
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR
        imageB = (imageB + (meanG - meanB) * (1 - imageB) * imageG) * maxB
    elif flag == 1:
        imageR = (imageR + (meanG - meanR) * (1 - imageR) * imageG) * maxR

    imageG *= maxG

    # Stack channels back together
    compensated_image = np.stack([imageR, imageG, imageB], axis=2).astype(np.uint8)
    return compensated_image

def gray_world(image):
    # Convert the image to a NumPy array
    image_array = np.array(image, dtype=np.float64)
    
    # Split into R, G, B channels
    imageR, imageG, imageB = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

    # Calculate channel means
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)

    # Mean gray value
    meanGray = np.mean((meanR + meanG + meanB) / 3)

    # Gray World Algorithm
    imageR *= (meanGray / meanR)
    imageG *= (meanGray / meanG)
    imageB *= (meanGray / meanB)

    # Stack, clip and return the white-balanced image
    white_balanced_image = np.clip(np.stack([imageR, imageG, imageB], axis=2), 0, 255).astype(np.uint8)
    return white_balanced_image

def sharpen(wbimage_array):
    # Apply Gaussian Blur to get smoothed image
    smoothed_array = cv2.GaussianBlur(wbimage_array, (0, 0), 1)  # Using OpenCV for blur

    # Perform unsharp masking
    sharpened_array = 2 * wbimage_array - smoothed_array

    # Clip the values to be in the valid range and convert back to uint8
    sharpened_image = np.clip(sharpened_array, 0, 255).astype(np.uint8)
    
    return sharpened_image


def load_model(model_path, num_classes=2, device='cpu'):
    """Load and prepare the model."""
    model = ssdlite320_mobilenet_v3_large(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True,  map_location=torch.device("cpu")))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path):
    """Load and preprocess image."""
    original_image = Image.open(image_path)
    compensated_image = compensate_RB(original_image, 0)
    white_balanced_image = gray_world(compensated_image)
    sharpened_image = sharpen(white_balanced_image)
    return sharpened_image

def predict(model, image, device='cpu', confidence_threshold=0.5):
    """Run inference and return filtered bounding boxes."""
    image_tensor = to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    filtered_boxes = boxes[scores > confidence_threshold]
    return filtered_boxes

def draw_boxes(image, boxes):
    """Draw bounding boxes on an image."""
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)
    return image_cv