import cv2
import numpy as np

# Load camera calibration data
data = np.load('results.npz')

def segment_color(frame, lower_bound, upper_bound):
    # Convert frame to HSV color space and create a mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

def find_contours(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def estimate_pose(frame, contours, camera_matrix, dist_coeffs):
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Define object points (3D points of an object in meters)
        phone_width = 0.0751 # meters
        phone_height = 0.159 # meters
        object_points = np.array([
            [-phone_width/2, -phone_height/2, 0],
            [phone_width/2, -phone_height/2, 0],
            [phone_width/2, phone_height/2, 0],
            [-phone_width/2, phone_height/2, 0]
        ], dtype=np.float32)
        
        # Define image points (2D points in the image plane)
        image_points = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)
        
        # SolvePnP to find pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )
        if success:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            distance = np.linalg.norm(translation_vector)
            # Calculate Euler angles from rotation matrix
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
            singular = sy < 1e-6
            if not singular:
                x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:  # Gimbal lock case
                x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                z_angle = 0
            # Convert angles from radians to degrees
            x_angle = np.degrees(x_angle)
            y_angle = np.degrees(y_angle)
            z_angle = np.degrees(z_angle)
            # Display distance and Euler angles on the frame
            cv2.putText(frame, f'Distance: {distance:.2f}m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Angles: X={x_angle:.2f}, Y={y_angle:.2f}, Z={z_angle:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def initialize_optical_flow(frame, mask, bounding_box, feature_params):
    x, y, w, h = bounding_box
    roi = mask[y:y+h, x:x+w]  # Region of Interest
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(roi_gray, mask=roi, **feature_params)
    if p0 is not None:
        p0 += np.array([[x, y]])  # Offset points by bounding box position
    return p0

def main():
    camera_matrix = data['mtx']
    dist_coeffs = data['dist']

    # Define lower and upper bounds for blue color in HSV
    lower_bound = np.array([100, 120, 50])
    upper_bound = np.array([130, 255, 255])

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    cap = cv2.VideoCapture(0)
    p0 = None
    old_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = segment_color(frame, lower_bound, upper_bound)
        contours = find_contours(mask)

        if contours and (p0 is None or len(p0) < 10):  # Initialize or reinitialize if features are lost
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = initialize_optical_flow(frame, mask, (x, y, w, h), feature_params)

        if p0 is not None:
            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
            old_gray = new_gray.copy()
            if p1 is not None:
                good_new = p1[st == 1]
                if len(good_new) > 5:  # Ensure there are enough features
                    x_min, y_min = np.min(good_new, axis=0).astype(int)
                    x_max, y_max = np.max(good_new, axis=0).astype(int)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    p0 = None  # Reinitialize
            else:
                p0 = None  # Reinitialize

        estimate_pose(frame, contours, camera_matrix, dist_coeffs)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()