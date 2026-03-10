import cv2
import cv2.aruco as aruco
import numpy as np

###################################
# CAMERA
###################################

cap = cv2.VideoCapture(0)

###################################
# ARUCO SETTINGS
###################################

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

###################################
# CAMERA CALIBRATION (simple test)
###################################

camera_matrix = np.array([[640,0,320],
                          [0,640,240],
                          [0,0,1]])

dist_coeff = np.zeros((5,1))

marker_size = 0.45   # marker size in meters

###################################
# MAIN LOOP
###################################

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    h, w = frame.shape[:2]
    center_x = w/2
    center_y = h/2

    if ids is not None:

        aruco.drawDetectedMarkers(frame, corners, ids)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners,
            marker_size,
            camera_matrix,
            dist_coeff
        )

        x = tvec[0][0][0]
        y = tvec[0][0][1]

        print("Offset X:",x," Offset Y:",y)

        ##################################
        # DIRECTION DECISION
        ##################################

        threshold = 0.05

        if x > threshold:
            print("MOVE RIGHT")

        elif x < -threshold:
            print("MOVE LEFT")

        if y > threshold:
            print("MOVE BACK")

        elif y < -threshold:
            print("MOVE FORWARD")

        if abs(x) < threshold and abs(y) < threshold:
            print("CENTERED - READY TO LAND")

    else:

        print("Marker not detected")

    ##################################
    # DISPLAY
    ##################################

    cv2.circle(frame,(int(center_x),int(center_y)),5,(0,0,255),-1)

    cv2.imshow("Aruco Detection",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
