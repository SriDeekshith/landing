import cv2
import cv2.aruco as aruco
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math

############################################
# CONNECT TO PIXHAWK
############################################

print("Connecting to Pixhawk")
vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True)

############################################
# CAMERA
############################################

cap = cv2.VideoCapture(0)

############################################
# ARUCO SETTINGS
############################################

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

marker_size = 0.45   # meters (your big marker)

############################################
# CAMERA CALIBRATION (basic placeholder)
############################################

camera_matrix = np.array([[640,0,320],
                          [0,640,240],
                          [0,0,1]])

dist_coeff = np.zeros((5,1))

############################################
# GPS TARGET LOCATION
############################################

target_lat = 16.506174
target_lon = 80.648015
target_alt = 5

############################################
# TAKEOFF FUNCTION
############################################

def arm_and_takeoff(aTargetAltitude):

    while not vehicle.is_armable:
        print("Waiting for vehicle")
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        time.sleep(1)

    vehicle.simple_takeoff(aTargetAltitude)

    while True:

        alt = vehicle.location.global_relative_frame.alt
        print("Altitude:", alt)

        if alt >= aTargetAltitude * 0.95:
            print("Reached altitude")
            break

        time.sleep(1)

############################################
# DISTANCE FUNCTION
############################################

def get_distance(aLocation1, aLocation2):

    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon

    return math.sqrt((dlat*dlat)+(dlong*dlong)) * 1.113195e5

############################################
# SEND VELOCITY
############################################

def send_velocity(vx,vy,vz):

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0,0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0,0,0,
        vx,vy,vz,
        0,0,0,
        0,0)

    vehicle.send_mavlink(msg)
    vehicle.flush()

############################################
# TAKEOFF
############################################

arm_and_takeoff(5)

############################################
# GO TO GPS LOCATION
############################################

print("Going to target location")

target = LocationGlobalRelative(target_lat,target_lon,target_alt)

vehicle.simple_goto(target)

while True:

    current = vehicle.location.global_relative_frame
    distance = get_distance(current,target)

    print("Distance to target:",distance)

    if distance < 2:
        print("Reached target location")
        break

    time.sleep(2)

############################################
# SEARCH FOR MARKER
############################################

print("Searching marker")

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    if ids is not None:

        print("Marker detected")

        rvec,tvec,_ = aruco.estimatePoseSingleMarkers(
            corners,
            marker_size,
            camera_matrix,
            dist_coeff
        )

        x = tvec[0][0][0]
        y = tvec[0][0][1]

        altitude = vehicle.location.global_relative_frame.alt

        print("Offset X:",x)
        print("Offset Y:",y)

        ###################################
        # CENTER CONTROL
        ###################################

        vx = -x * 0.5
        vy = -y * 0.5

        ###################################
        # DESCEND
        ###################################

        if altitude > 1.5:
            vz = 0.3
        else:
            vz = 0

        send_velocity(vx,vy,vz)

        ###################################
        # FINAL LAND
        ###################################

        if abs(x) < 0.05 and abs(y) < 0.05 and altitude < 1:

            print("Landing on marker")

            vehicle.mode = VehicleMode("LAND")

            break

    else:

        print("Searching marker")

        send_velocity(0,0,0)

    cv2.imshow("camera",frame)

    if cv2.waitKey(1) == 27:
        break

############################################
# CLEANUP
############################################

cap.release()
cv2.destroyAllWindows()
vehicle.close()
