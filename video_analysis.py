import cv2
import numpy as np
import pandas as pd
import dlib
from math import hypot

video_name = '<video_file_name>' # Video file name without extension (e.g. mov)

cap = cv2.VideoCapture(video_name + '.mov')
_, frame = cap.read()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

FONT = cv2.FONT_HERSHEY_PLAIN


def midpoint(pt1, pt2):
    return int((pt1.x + pt2.x)/2), int((pt1.y + pt2.y)/2)


def get_blinking_ratio(facial_landmarks, eye_points):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    centre_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    centre_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    # CHECKING RATIO OF HORIZONTAL AND VERTICAL LINES USING EUCLIDEAN DISTANCE
    horizontal_line_length = hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])
    vertical_line_length = hypot(centre_top[0] - centre_bottom[0], centre_top[1] - centre_bottom[1])
    
    # AVOIDING DIVISION BY ZERO ERROR
    if vertical_line_length != 0:
        ratio = horizontal_line_length / vertical_line_length
    else:
        ratio = 5
        
    return ratio


def get_gaze_ratio(facial_landmarks, eye_points):
    # ISOLATING EYE REGION
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                          np.int32)

    eye = gray
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # THRESHOLDING EYE REGION TO ISOLATE IRIS
    if max_y - min_y > 0 and max_x - min_x > 0:
        gray_eye = eye[min_y:max_y, min_x:max_x]
        cv2.imshow("Eye", gray_eye)
     
        _, threshold_eye = cv2.threshold(gray_eye, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        threshold_eye = cv2.erode(threshold_eye, None, iterations=2)
        threshold_eye = cv2.dilate(threshold_eye, None, iterations=4)
        
        # DIVIDING THRESHOLDED REGION IN HALF TO ALLOW FOR GAZE DETECTION
        height, width = threshold_eye.shape
        
        left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
        right_side_threshold = threshold_eye[0:height, int(width / 2):width]
    
        # COUNTING NUMBER OF WHITE PIXELS IN EACH HALF TO DETERMINE WHICH CONTAINS MORE IRIS
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_white = cv2.countNonZero(right_side_threshold)
    
        cv2.imshow("Threshold", threshold_eye)
        cv2.imshow("Left", left_side_threshold)
        cv2.imshow("Right", right_side_threshold)
    
        # AVOIDING ERROR IF EYE NOT FOUND (E.G. IF EYE CLOSED)
        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 3
        else:
            gaze_ratio = left_side_white / right_side_white
    else:
        gaze_ratio = 1.5
    
    return gaze_ratio


gaze_df = pd.DataFrame(columns=['Frame', 'Direction', 'Rotation Vector','Timestamp'])
frame_count = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        size = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            # DRAWING RECTANGLE AROUND FACE
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)

            # DETECTING BLINKING
            landmarks = predictor(gray, face)
            left_eye_ratio = get_blinking_ratio(landmarks, [36, 37, 38, 39, 40, 41])
            right_eye_ratio = get_blinking_ratio(landmarks, [42, 43, 44, 45, 46, 47])
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2


            # DETECTING GAZE DIRECTION
            gaze_ratio_left = get_gaze_ratio(landmarks, [36, 37, 38, 39, 40, 41])
            gaze_ratio_right = get_gaze_ratio(landmarks, [42, 43, 44, 45, 46, 47])
            gaze_ratio = (gaze_ratio_left + gaze_ratio_right) / 2


            # INDICATING GAZE DIRECTION            
            if blinking_ratio >= 5:
                cv2.putText(frame, "BLINKING", (50, 100), FONT, 2, (0, 0, 255), 3)
                gaze_direction = "Blinking"
            else:
                if gaze_ratio < 1:
                    cv2.putText(frame, "LEFT", (50, 150), FONT, 2, (0, 0, 255), 3)
                    gaze_direction = "Left"
                elif 1 < gaze_ratio < 2:
                    cv2.putText(frame, "GAZE_AWAY", (50, 150), FONT, 2, (0, 0, 255), 3)
                    gaze_direction = "Gaze_Away"
                else:
                    cv2.putText(frame, "RIGHT", (50, 150), FONT, 2, (0, 0, 255), 3)
                    gaze_direction = "Right"
           
            #2D IMAGE POINTS
            image_points = np.array([
                            (landmarks.part(33).x, landmarks.part(33).y),     # Nose tip
                            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
                            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
                            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corne
                            (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
                            (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
                        ], dtype="double")
 
            # 3D MODEL POINTS
            model_points = np.array([
                                        (0.0, 0.0, 0.0),             # Nose tip
                                        (0.0, -330.0, -65.0),        # Chin
                                        (-225.0, 170.0, -135.0),     # Left eye left corner
                                        (225.0, 170.0, -135.0),      # Right eye right corne
                                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                                        (150.0, -150.0, -125.0)      # Right mouth corner
                                     
                                    ])
             
             
            # CAMERA INTERNALS
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                                     [[focal_length, 0, center[0]],
                                     [0, focal_length, center[1]],
                                     [0, 0, 1]], dtype = "double"
                                     )
             
             
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            # PROJECTING LINE INDICATING HEAD ORIENTATION
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
             
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
             
            cv2.line(frame, p1, p2, (255,0,0), 2)
            
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            gaze_df = gaze_df.append({'Frame': frame_count, 'Direction': gaze_direction, 'Rotation Vector': ','.join(str(x) for x in rotation_vector), 'Timestamp': timestamp}, ignore_index=True)
            frame_count += 1
    
            cv2.imshow("Frame", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


cap.release()
cv2.destroyAllWindows()

csv_name = video_name + "_df.csv"
gaze_df.to_csv(csv_name, index=False)
