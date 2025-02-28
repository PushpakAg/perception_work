import cv2,torch,os,numpy as np,math
from utils import *

def mask_red_green(image):
    mask=cv2.inRange(image,np.array([0,0,100]),np.array([50,50,255]))
    mask|=cv2.inRange(image,np.array([0,10,0]),np.array([50,255,50]))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((15,15)))
    return cv2.dilate(mask,None,iterations=3)

def rotationMatrixToEulerAngles(R):
    sy=np.sqrt(R[0,0]**2 + R[1,0]**2)
    x,y,z=np.arctan2(R[2,1],R[2,2]),np.arctan2(-R[2,0],sy),np.arctan2(R[1,0],R[0,0])
    return np.degrees([x,y,z])

def estimate_pose(object_pts,image_pts,K,d):
    assert len(image_pts)>=4
    success,rvec,tvec,inliers=cv2.solvePnPRansac(object_pts,image_pts,K,d,flags=cv2.SOLVEPNP_ITERATIVE,reprojectionError=8.0,confidence=0.99,iterationsCount=100)
    if not success or len(inliers)<4:return None
    return rvec,tvec,inliers
def process_images(model, img_dir, gw, gh, K, d, device='cuda',
                   min_bbox_size=(50,50), aspect_ratio_thresh=0.5,
                   min_keypoints=20, min_hull_area=1000, smoothing_factor=0.3):

    orb = cv2.ORB_create(500)
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    prev_bbox = None
    prev_pose = None 

    for file in files:
        orig = preprocess_image(os.path.join(img_dir, file))
        frame = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        detections = predict(model, orig, device)

        if detections.size == 0:
            print("No detections. Skipping Frame.")
            cv2.waitKey(1)
            continue

        x1, y1, x2, y2 = detections[0].astype(int)

        bbox_w, bbox_h = x2 - x1, y2 - y1
        if bbox_w < min_bbox_size[0] or bbox_h < min_bbox_size[1]:
            print(f"Skipping frame (bounding box too small: {bbox_w}x{bbox_h})")
            cv2.waitKey(1)
            continue  

        bbox_aspect_ratio = bbox_w / float(bbox_h)
        expected_AR = gw / gh
        if abs(np.log(bbox_aspect_ratio / expected_AR)) > aspect_ratio_thresh:
            print(f"Skipping frame due to unusual AR: {bbox_aspect_ratio:.2f}")
            cv2.waitKey(1)
            continue

        current_bbox = np.array([x1,y1,x2,y2], np.float32)
        if prev_bbox is not None:
            smoothed_bbox = smoothing_factor*current_bbox + (1-smoothing_factor)*prev_bbox
        else:
            smoothed_bbox = current_bbox

        prev_bbox = smoothed_bbox
        x1,y1,x2,y2 = smoothed_bbox.astype(int)

        pad_x, pad_y = int((x2 - x1)*0.2), int((y2 - y1)*0.2)
        x1, y1 = max(x1-pad_x,0), max(y1-pad_y,0)
        x2, y2 = min(x2+pad_x,frame.shape[1]), min(y2+pad_y,frame.shape[0])

        crop = frame[y1:y2, x1:x2]

        mask = mask_red_green(crop)
        kp, des = orb.detectAndCompute(mask, None)

        vis = cv2.drawKeypoints(crop.copy(), kp, None, (0,255,0), flags=0)

        if len(kp) < min_keypoints:
            print(f"Too few keypoints ({len(kp)}). Skipping frame.")
            if prev_pose:
                rvec, tvec = prev_pose
                R, _ = cv2.Rodrigues(rvec)
                angles = rotationMatrixToEulerAngles(R)
                dist = np.linalg.norm(tvec)
                print(f"[Using last pose] Dist:{dist:.2f}|Angle[PitchYawRoll]:{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f}")
            cv2.imshow('features', vis)
            cv2.waitKey(1)
            continue 

        pts = np.float32([p.pt for p in kp])
        hull = cv2.convexHull(pts).reshape(-1,2)

        hull_area = cv2.contourArea(hull)
        if len(hull) < 4 or hull_area < min_hull_area:
            print(f"Convex hull insufficient (size:{len(hull)}, area:{hull_area:.2f}). Skipping.")
            cv2.imshow('features', vis)
            cv2.waitKey(1)
            continue  
        rect = cv2.minAreaRect(hull)
        image_box = cv2.boxPoints(rect)

        object_pts = np.array([[0,0,0],[gw,0,0],[gw,gh,0],[0,gh,0]], np.float32)

        image_pts = np.array(sorted(image_box, key=lambda x:(x[1],x[0])), np.float32)
        top, bottom = sorted(image_pts[:2], key=lambda x:x[0]), sorted(image_pts[2:], key=lambda x:x[0])
        image_pts = np.array([top[0],top[1],bottom[1],bottom[0]], np.float32)

        res = estimate_pose(object_pts, image_pts, K, d)

        if res is not None:
            rvec, tvec, inliers = res
            prev_pose = (rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            angles = rotationMatrixToEulerAngles(R)
            dist = np.linalg.norm(tvec)
            print(f'Distance:{dist:.2f}m | Angle[Pitch,Yaw,Roll]:{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f}')

            for i in image_pts.astype(int):
                cv2.circle(vis, tuple(i),8,(0,0,255),-1)
            cv2.polylines(vis, [image_pts.astype(int)],True,(255,0,0),2)
        else:
            print("Pose estimation failed. Using previous pose (if available).")
            if prev_pose:
                rvec, tvec = prev_pose
                R,_ = cv2.Rodrigues(rvec)
                angles = rotationMatrixToEulerAngles(R)
                dist = np.linalg.norm(tvec)
                print(f"[Reuse stable pose] Dist:{dist:.2f}|Angle[PitchYawRoll]:{angles[0]:.2f},{angles[1]:.2f},{angles[2]:.2f}")

        cv2.imshow('features', vis)
        cv2.imshow('mask', mask)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break
if __name__=="__main__":
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cuda')
    model=load_model('models/ssd_mobilenetv3_single_class.pth',device=device)
    image_directory="/home/pushpak/Downloads/data 3/" # modify as needed
    
    data = np.load('/home/pushpak/Desktop/gate_mobilenet/MultiMatrix.npz')
    camera_matrix=data["camMatrix"]
    dist_coeffs=data["distCoef"]

    gate_width = 1.5
    gate_height = 1
    process_images(model,image_directory,gate_width,gate_height,camera_matrix,dist_coeffs,device=device)