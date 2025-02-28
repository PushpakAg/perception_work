
import os
import cv2
import numpy as np
import torch
from utils import *

def mask_red_green(image):
    lower_red = np.array([0, 0, 100]) 
    upper_red = np.array([50, 50, 255])
    lower_green = np.array([0, 10, 0]) 
    upper_green = np.array([50, 255, 50])
    red_mask = cv2.inRange(image, lower_red, upper_red)
    green_mask = cv2.inRange(image, lower_green, upper_green)
    
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    
    return combined_mask

def process_images_from_directory(model, image_directory, device='cpu'):
    """Process all images in a given directory."""
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the directory.")
        return
    image_files.sort()
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Masked Regions', cv2.WINDOW_NORMAL)
    
    for img_file in image_files:
        image_path = os.path.join(image_directory, img_file)
        image = preprocess_image(image_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes = predict(model, image, device)

        image_with_boxes = draw_boxes(image, boxes)

        for (x_min, y_min, x_max, y_max) in boxes:
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cropped_region = image_bgr[y_min:y_max, x_min:x_max]
            masked_region = mask_red_green(cropped_region)

        
        cv2.imshow('Video', image_with_boxes)
        cv2.imshow('Masked Regions', masked_region)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_path = 'models/ssd_mobilenetv3_single_class.pth'
    image_directory  = "/home/pushpak/Downloads/data 3/"
    model = load_model(model_path, device=device)
    process_images_from_directory(model, image_directory, device=device)