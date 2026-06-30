import cv2
import numpy as np
import src.primary.config as config
from src.primary.geometry import estimateTargetWorldPosition

class Detection:
    def __init__(self, u: float | None, v: float | None, px_w: float | None, px_h: float | None):
        self.u = u 
        self.v = v
        self.px_w = px_w 
        self.px_h = px_h 

class Measurement:
    def __init__(self, x: float | None, y: float | None, z: float | None = None, 
                 pitch: float | None = None, roll: float | None = None, yaw: float | None = None):
                
        self.x = x # x points right
        self.y = y # y points down
        self.z = z # z points away from the camera
        
        #probably not used:
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw 

# gold medal
## LOWER_HSV = np.array([14, 50, 100])
## UPPER_HSV = np.array([25, 130, 200])
# LOWER_HSV = np.array([13, 45, 85])
# UPPER_HSV = np.array([27, 160, 240])

#tennis ball
LOWER_HSV = np.array([23, 35, 110])
UPPER_HSV = np.array([40, 220, 255])

MIN_CONTOUR_AREA = 100

def detectSingleObject(frame):
    
    #implementation TBD#################
    # consider using minAreaRect
    # need adjsut hsv range and filter strategy as needed

    # Classical CV Detector using thresholding
    u, v, px_w, px_h = None, None, None, None
    x, y, z, pitch, roll, yaw = None, None, None, None, None, None
    object_detected = False

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(frame, LOWER_HSV, UPPER_HSV) # Color threshold
    mask = cv2.medianBlur(mask, 5) # apply blur
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # remove small random white specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # fil small black holes/gaps 

    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0: 
        largest_contour = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > MIN_CONTOUR_AREA:
            u, v, px_w, px_h = cv2.boundingRect(largest_contour)
            u = u + px_w/2
            v = v + px_h/2

            x, y, z = estimateTargetWorldPosition(u, v, px_w, px_h)
            object_detected = True

    detection = Detection(u, v, px_w, px_h)
    measurement = Measurement(x,y,z,pitch,roll,yaw) 
    return object_detected, detection, measurement




