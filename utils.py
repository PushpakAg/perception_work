import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import to_tensor
import torch
import math


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




#------------------------------Pose Estimation SetUp--------------------------

data = np.load("MultiMatrix.npz")
camMatrix = data['camMatrix']
distCoef = data['distCoef']

# Extract focal length (assuming fx and fy are equal)
FOCAL_LENGTH_PIXELS = camMatrix[0, 0]
IMAGE_WIDTH = camMatrix[0, 2] * 2  # cx = IMAGE_WIDTH / 2

# Calculate Horizontal Field of View (HFOV)
HFOV = 2 * np.degrees(np.arctan((IMAGE_WIDTH / (2 * FOCAL_LENGTH_PIXELS))))

# Calculate Sensor Width using FOV (in meters)
FOCAL_LENGTH_METERS = FOCAL_LENGTH_PIXELS / IMAGE_WIDTH  # Focal length proportionate to sensor width
SENSOR_WIDTH = 2 * FOCAL_LENGTH_METERS * math.tan(math.radians(HFOV / 2))

# Conversion factor (how many pixels per meter)
meters_to_pixels = FOCAL_LENGTH_PIXELS / SENSOR_WIDTH
pixels_to_meters = 1 / meters_to_pixels  # To convert pixels back to meters


# Define rectangle dimensions in meters
rectangle_length_m = 1.4  # Example length in meters
rectangle_breadth_m = 1  # Example breadth in meters

rectangle_length_px = rectangle_length_m * meters_to_pixels
rectangle_breadth_px = rectangle_breadth_m * meters_to_pixels

model_points = np.array([
    [0.0, 0.0, 0.0],  # Point 1 (bottom-left corner)
    [rectangle_length_px, 0.0, 0.0],  # Point 2 (bottom-right corner)
    [rectangle_length_px, rectangle_breadth_px, 0.0],  # Point 3 (top-right corner)
    [0.0, rectangle_breadth_px, 0.0]  # Point 4 (top-left corner)
], dtype=np.float32)

# Centered at Y-axis
camera_matrix = camMatrix

# Get from calibration file
dist_coeffs = distCoef




#-----------------------------------------------------------------------------




def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles (yaw, pitch, roll).
    The angles are returned in degrees.
    """
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    # Check if near singularity to avoid gimbal lock issues
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])  # Roll
        y = math.atan2(-rotation_matrix[2, 0], sy)                   # Pitch
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])  # Yaw
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])
def estimate_pose(points):
    if len(points) != 4:
        print("4 points required")
        return None

    image_points = np.array(points, dtype=np.float32)

    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        print("Pose estimation failed")
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract roll, pitch, and yaw
    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)

    # Distance from the camera to the rectangle
    distance_in_pixels = np.linalg.norm(translation_vector)
    distance_in_meters = distance_in_pixels * pixels_to_meters

    if(yaw<-90):
        yaw= -(180+yaw)
    else:
        yaw= (180-yaw)

    print(f"Pose estimation successful!")
    print(f"Roll: {roll:.2f} degrees")
    print(f"Pitch: {pitch:.2f} degrees")
    print(f"Yaw: {yaw:.2f} degrees")
    print(f"Distance: {distance_in_meters:.2f} meters")

    return roll, pitch, yaw, distance_in_meters






#---------------------------Hough Lines and Filters------------------------------------------------



def Filters(image):
    image_copy2 = image.copy()
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab_image)


    clahe_for_l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    clahe_for_a = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    clahe_for_b = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

    clahe_b = clahe_for_b.apply(b_channel)
    clahe_a = clahe_for_a.apply(a_channel)
    clahe_l = clahe_for_l.apply(l_channel)

    lab_clahe = cv2.merge((clahe_l, clahe_a, clahe_b))

    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # cv2.imshow("Mid",balanced_image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    
    exposure_factor = (-0.0044117) * brightness + 1.695287

    # balanced_image = np.clip(balanced_image * exposure_factor, 0, 255).astype(np.uint8)
    balanced_image = image
    scale = 1.2
    delta = 0
    ddepth = cv2.CV_16S
    blurred_image = cv2.GaussianBlur(balanced_image, (3, 3), 0)
    gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
  
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

  
    # cv2.imshow("Segment", grad)

    _, grad = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)

    linesP = cv2.HoughLinesP(grad, 1, np.pi / 180, 30, None, 75, 10)

    if linesP is not None:
        extended_lines = []

        for i in range(0, len(linesP)):
            l = linesP[i][0]

            dx = l[2] - l[0]
            dy = l[3] - l[1]

            length = np.sqrt(dx**2 + dy**2)

            direction = (dx / length, dy / length)

            extend_length = 100
            new_x1 = int(l[0] - direction[0] * extend_length)
            new_y1 = int(l[1] - direction[1] * extend_length)
            new_x2 = int(l[2] + direction[0] * extend_length)
            new_y2 = int(l[3] + direction[1] * extend_length)

            cv2.line(image_copy2, (new_x1, new_y1), (new_x2, new_y2), (0, 0, 255), 3, cv2.LINE_AA)

            extended_lines.append([[new_x1, new_y1, new_x2, new_y2], direction])

        intersections = []
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):

                (x1, y1, x2, y2), dir1 = extended_lines[i]
                (x3, y3, x4, y4), dir2 = extended_lines[j]

                #Line 1
                a1 = y2 - y1
                b1 = x1 - x2
                c1 = a1 * x1 + b1 * y1

                #Line 2 
                a2 = y4 - y3
                b2 = x3 - x4
                c2 = a2 * x3 + b2 * y3

                #determinant
                det = a1 * b2 - a2 * b1

                if det != 0:
                    x = (b2 * c1 - b1 * c2) / det
                    y = (a1 * c2 - a2 * c1) / det
                    intersection = (int(x), int(y))

                    cos_theta = np.dot(dir1, dir2)

                    # Convert to an angle in degrees
                    angle = np.degrees(np.arccos(cos_theta))

                    if 70 < angle < 110:
                        intersections.append(intersection)
                        ###cv2.circle(image_copy2, intersection, 5, (0, 255, 0), -1)

        def distance_between(point1, point2):
            return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        cluster_threshold = 2500    # 50 squared

        if intersections:
            clusters = []
            used_points = set()

            for i, point1 in enumerate(intersections):
                if i in used_points:
                    continue
                cluster = [point1]
                used_points.add(i)

                for j, point2 in enumerate(intersections):
                    if j in used_points:
                        continue
                    if distance_between(point1, point2) < cluster_threshold:
                        cluster.append(point2)
                        used_points.add(j)

                Min_Cluster_Size = 20
                if len(cluster) > Min_Cluster_Size:
                    avg_x = int(np.mean([p[0] for p in cluster]))
                    avg_y = int(np.mean([p[1] for p in cluster]))
                    avg_point = (avg_x, avg_y)

                    clusters.append(avg_point)

            for point in clusters:
                cv2.circle(image_copy2, point, 5, (0, 255, 255), -1)

            estimate_pose(clusters)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", image_copy2)
    return image_copy2