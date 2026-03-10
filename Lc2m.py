import cv2
import cv2.aruco as aruco
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math

############################################
# CONNECT PIXHAWK
############################################

vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True)

############################################
# CAMERA
############################################

cap = cv2.VideoCapture(0)

############################################
# ARUCO
############################################

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

BIG_MARKER_SIZE = 0.45
SMALL_MARKER_SIZE = 0.18

############################################
# CAMERA CALIBRATION
############################################

camera_matrix = np.array([[640,0,320],
                          [0,640,240],
                          [0,0,1]])

dist_coeff = np.zeros((5,1))

############################################
# GPS TARGET
############################################

target_lat = 16.506174
target_lon = 80.648015
target_alt = 6

############################################
# PID CONTROLLER
############################################

class PID:

    def __init__(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self,error):

        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        return self.kp*error + self.ki*self.integral + self.kd*derivative

pid_x = PID(0.5,0.01,0.2)
pid_y = PID(0.5,0.01,0.2)

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

def arm_and_takeoff(alt):

    while not vehicle.is_armable:
        time.sleep(1)

    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        time.sleep(1)

    vehicle.simple_takeoff(alt)

    while vehicle.location.global_relative_frame.alt < alt*0.95:
        time.sleep(1)

############################################
# SPIRAL SEARCH
############################################

def spiral_search():

    print("Marker not found → spiral search")

    radius = 0.5
    angle = 0

    for i in range(60):

        vx = radius * math.cos(angle)
        vy = radius * math.sin(angle)

        send_velocity(vx,vy,0)

        angle += 0.3
        radius += 0.02

        time.sleep(0.5)

############################################
# DISTANCE CHECK
############################################

def distance_to_target(current,target):

    dlat = target.lat - current.lat
    dlon = target.lon - current.lon

    return math.sqrt((dlat*dlat)+(dlon*dlon))*1.113195e5

############################################
# TAKEOFF
############################################

arm_and_takeoff(6)

############################################
# FLY TO GPS
############################################

target = LocationGlobalRelative(target_lat,target_lon,target_alt)

vehicle.simple_goto(target)

while True:

    current = vehicle.location.global_relative_frame

    if distance_to_target(current,target) < 2:
        break

    time.sleep(1)

print("Reached GPS target")

############################################
# MARKER LANDING LOOP
############################################

while True:

    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners,ids,rejected = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)

    altitude = vehicle.location.global_relative_frame.alt

    ########################################
    # MARKER SIZE SWITCH
    ########################################

    if altitude > 2:
        marker_size = BIG_MARKER_SIZE
    else:
        marker_size = SMALL_MARKER_SIZE

    ########################################
    # IF MARKER FOUND
    ########################################

    if ids is not None:

        rvec,tvec,_ = aruco.estimatePoseSingleMarkers(
            corners,
            marker_size,
            camera_matrix,
            dist_coeff)

        x = tvec[0][0][0]
        y = tvec[0][0][1]

        ####################################
        # PID CONTROL
        ####################################

        vx = -pid_x.compute(x)
        vy = -pid_y.compute(y)

        ####################################
        # DESCEND
        ####################################

        if altitude > 1.5:
            vz = 0.25
        else:
            vz = 0

        send_velocity(vx,vy,vz)

        ####################################
        # FINAL LAND
        ####################################

        if abs(x)<0.03 and abs(y)<0.03 and altitude<1:

            print("Precision landing")

            vehicle.mode = VehicleMode("LAND")

            break

    ########################################
    # MARKER NOT FOUND
    ########################################

    else:

        spiral_search()

    cv2.imshow("camera",frame)

    if cv2.waitKey(1)==27:
        break

############################################
# CLEANUP
############################################

cap.release()
cv2.destroyAllWindows()
vehicle.close()
