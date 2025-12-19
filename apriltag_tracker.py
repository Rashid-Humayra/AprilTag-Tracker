import cv2
import pyapriltags#apriltag
import numpy as np
import os



calibration_path = 'C:/Users/hrzan/Documents/NTNU 1st Semester/Specialization Project/camera-calibration/output'

dist_coeffs = np.loadtxt(os.path.join(calibration_path, 'distortion_coefficients.txt'), dtype=np.float32)
dist_coeffs = dist_coeffs.reshape(-1)


# tag parameters
tag_size = 0.22  # meters
tag_ids = [0, 1, 2] #id 0 in the middle, id 1 on the right and id 2 on the left

# defining the 3D corners of the tag in its local frame
object_points = np.array([
    [-tag_size/2, -tag_size/2, 0],
    [ tag_size/2, -tag_size/2, 0],
    [ tag_size/2,  tag_size/2, 0],
    [-tag_size/2,  tag_size/2, 0]
], dtype=np.float32)




cap = cv2.VideoCapture(0) # camera initialization

target_width = 1920
target_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)




# Camera calibration
# fx = 1500#600   focal length x (pixels)
# fy = 1500#600   focal length y (pixels)
# cx = actual_w/2.0  #320   optical center x
# cy = actual_h/2.0  #240   optical center y


camera_matrix = np.loadtxt(os.path.join(calibration_path, 'camera_matrix.txt'), dtype=np.float32)

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]], dtype=np.float32)



detector = pyapriltags.Detector(families="tag36h11")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    
    ###### updates adding new logic
    tag_poses = {}
    

    for det in detections:
        tag_id = det.tag_id
        if tag_id not in tag_ids:
            continue

        # this is to get 3D pose
        image_points = det.corners.astype(np.float32)
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        x, y, z = tvec.flatten()
        
        if ret:
            tag_poses[tag_id] = {'rvec': rvec, 'tvec': tvec}
            # for drawing the tag outline
            for i in range(4):
                pt1 = tuple(det.corners[i].astype(int))
                pt2 = tuple(det.corners[(i+1)%4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            
            
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
            
            
            z = tvec.flatten()[2]
            cv2.putText(frame, f"ID: {tag_id} Z: {z:.2f} m", 
                        (int(det.center[0]), int(det.center[1]-30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        
        # horizontal_distance = abs(x - x_line)

        # angle_rad = np.arctan2(horizontal_distance, z)
        # angle_deg = np.degrees(angle_rad)

        # cv2.putText(frame, f"ID: {det.tag_id}", (int(det.center[0]), int(det.center[1]-30)),
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        # cv2.putText(frame, f"Horiz Dist: {horizontal_distance:.2f} m", (10, 30),
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        # cv2.putText(frame, f"Angle: {angle_deg:.2f} deg", (10, 60),
        #              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        


    if 1 in tag_poses and 2 in tag_poses:
        tvec_1 = tag_poses[1]['tvec'].flatten()
        tvec_2 = tag_poses[2]['tvec'].flatten()
        
        # Lateral X-Error (Midpoint between Tag 1 and Tag 2)
        X_Midpoint_Error = (tvec_1[0] + tvec_2[0]) / 2
        
        # Average approach distance (Z)
        Z_Entrance_Avg = (tvec_1[2] + tvec_2[2]) / 2
        

        cv2.putText(frame, f"S1: X-Midpoint-Error: {X_Midpoint_Error:.2f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"S1: Z-Entry-Dist: {Z_Entrance_Avg:.2f} m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # Checking if the final center tag (0) is present for Stage 3 Precision
    if 0 in tag_poses:
        tvec_0 = tag_poses[0]['tvec'].flatten()
        rvec_0 = tag_poses[0]['rvec'].flatten()
        
        
        # Yaw_Target_Error = rvec_2[2] 
        
        # cv2.putText(frame, f"S3: X-Target: {tvec_2[0]:.2f} m", (10, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, f"S3: Z-Target: {tvec_2[2]:.2f} m", (10, 130),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, f"S3: Yaw Error: {Yaw_Target_Error:.2f} rad", (10, 160),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        x = tvec_0[0]  # X-coordinate of the Middle Tag
        z = tvec_0[2]  # Z-coordinate of the Middle Tag
        x_line = 0     # Assuming the 'floor vertical line' is the camera's X=0 axis
        
    
        horizontal_distance = abs(x - x_line)

        # angle calculation
        angle_rad = np.arctan2(x, z)
        angle_deg = np.degrees(angle_rad)
        
        
        cv2.putText(frame, f"S3: Lateral Error (X): {x:.2f} m", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"S3: Approach Dist (Z): {z:.2f} m", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"S3: Horizontal Dist: {horizontal_distance:.2f} m", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"S3: Azimuth Angle: {angle_deg:.2f} deg", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        # Yaw_Target_Error = rvec_0[2] 
        # cv2.putText(frame, f"S3: Yaw Error: {Yaw_Target_Error:.2f} rad", (10, 190),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    
    
   
        
    cv2.imshow("AprilTag Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
