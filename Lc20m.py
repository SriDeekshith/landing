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

print("Connecting Pixhawk...")
vehicle = connect('/dev/ttyAMA0', baud=921600, wait_ready=True)

############################################
# CAMERA
############################################

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

############################################
# ARUCO SETUP
############################################

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

############################################
# CAMERA CALIBRATION
############################################

camera_matrix = np.array([[900,0,640],
                          [0,900,360],
                          [0,0,1]])

dist_coeff = np.zeros((5,1))

############################################
# MARKER SIZES
############################################

MARKER_SIZES = {
0:1.2,   # big marker (20m)
1:0.6,   # medium marker
2:0.25   # small marker
}

############################################
# GPS TARGET
############################################

TARGET_LAT = 16.565772
TARGET_LON = 80.521778
TARGET_ALT = 20

############################################
# PID CONTROLLER
############################################

class PID:

    def __init__(self,kp,ki,kd):
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.prev=0
        self.integral=0

    def update(self,error):

        self.integral+=error
        derivative=error-self.prev
        self.prev=error

        return self.kp*error + self.ki*self.integral + self.kd*derivative

pid_x = PID(0.5,0.01,0.2)
pid_y = PID(0.5,0.01,0.2)

############################################
# VELOCITY COMMAND
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

def arm_takeoff(alt):

    while not vehicle.is_armable:
        print("Waiting vehicle")
        time.sleep(1)

    vehicle.mode=VehicleMode("GUIDED")
    vehicle.armed=True

    while not vehicle.armed:
        time.sleep(1)

    print("Taking off")

    vehicle.simple_takeoff(alt)

    while True:

        h=vehicle.location.global_relative_frame.alt
        print("Altitude:",h)

        if h>=alt*0.95:
            break

        time.sleep(1)

############################################
# DISTANCE FUNCTION
############################################

def distance(a,b):

    dlat=b.lat-a.lat
    dlon=b.lon-a.lon

    return math.sqrt((dlat*dlat)+(dlon*dlon))*1.113195e5

############################################
# SPIRAL SEARCH
############################################

def spiral_search():

    print("Searching marker...")

    radius=0.4
    angle=0

    for i in range(40):

        vx=radius*math.cos(angle)
        vy=radius*math.sin(angle)

        send_velocity(vx,vy,0)

        angle+=0.5
        radius+=0.03

        time.sleep(0.4)

############################################
# TAKEOFF
############################################

arm_takeoff(TARGET_ALT)

############################################
# GO TO GPS TARGET
############################################

print("Flying to GPS location")

target = LocationGlobalRelative(TARGET_LAT,TARGET_LON,TARGET_ALT)

vehicle.simple_goto(target)

while True:

    current=vehicle.location.global_relative_frame

    dist=distance(current,target)

    print("Distance:",dist)

    if dist<2:
        break

    time.sleep(2)

print("Arrived GPS point")

############################################
# LANDING LOOP
############################################

while True:

    ret,frame=cap.read()

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners,ids,rejected=aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    altitude=vehicle.location.global_relative_frame.alt

    if ids is not None:

        for i in range(len(ids)):

            marker_id=int(ids[i])

            if marker_id not in MARKER_SIZES:
                continue

            marker_size=MARKER_SIZES[marker_id]

            rvec,tvec,_=aruco.estimatePoseSingleMarkers(
                corners[i],
                marker_size,
                camera_matrix,
                dist_coeff
            )

            x=tvec[0][0][0]
            y=tvec[0][0][1]

            ####################################
            # PID STABILIZATION
            ####################################

            vx=-pid_x.update(x)
            vy=-pid_y.update(y)

            ####################################
            # DESCEND
            ####################################

            if altitude>3:
                vz=0.4
            elif altitude>1:
                vz=0.2
            else:
                vz=0

            send_velocity(vx,vy,vz)

            ####################################
            # FINAL LAND
            ####################################

            if abs(x)<0.02 and abs(y)<0.02 and altitude<0.8:

                print("Landing")

                vehicle.mode=VehicleMode("LAND")

                break

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
