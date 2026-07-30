import cv2
import numpy as np
import src.primary.config as config
from src.primary.geometry import estimateObjectWorldPosition
from enum import Enum, auto
from src.primary.object_vision_spec import OBJECT_VISION_SPECS, ObjectType, ObjectVisionSpec
from src.primary.color import COLOR_SPECS

class TriangleDetection:
    def __init__(
        self, 
        vertices_px: list[list[float]] | np.ndarray, 
        color_id: str | None = None,
    ):
        self.vertices_px = np.asarray(vertices_px, dtype=np.float64)
        self.color_id = color_id


class Detection:
    def __init__(
        self, 
        u: float | None, v: float | None, px_w: float | None, px_h: float | None,
        triangles: list[TriangleDetection] | None = None,
    ):        
        self.u = u 
        self.v = v
        self.px_w = px_w 
        self.px_h = px_h 
        self.triangles = triangles if triangles is not None else []


class Measurement:
    def __init__(
        self, 
        x: float | None, y: float | None, z: float | None = None, 
        pitch: float | None = None, roll: float | None = None, yaw: float | None = None
    ):
                
        self.x = x # x points right
        self.y = y # y points down
        self.z = z # z points away from the camera
        
        #probably not used:
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw 


def detectSingleObject(frame: np.ndarray, object_type: ObjectType) -> tuple[bool, Detection, Measurement]:

    object_vision_spec = OBJECT_VISION_SPECS[object_type]

    if object_type == ObjectType.TENNIS_BALL:
        return detectTennisBall(frame, object_vision_spec,)

    elif object_type == ObjectType.PAPER_PLANE_TRIANGLES:
        return detectPaperPlaneTriangles(frame, object_vision_spec,)

    # elif object_type == ObjectType.PAPER_PLANE_ARUCO:
    # elif...

    raise ValueError(f"Unsupported object type: {object_type}")


def detectTennisBall(frame: np.ndarray, object_vision_spec: ObjectVisionSpec,) -> tuple[bool, Detection, Measurement]:

    detection = findSingleObjectUsingLargestColorBlob(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    x, y, z = estimateObjectWorldPosition(detection.u, detection.v, detection.px_w, detection.px_h, object_w=object_vision_spec.width)
    measurement = Measurement(x, y, z, None, None, None,)

    return True, detection, measurement


def detectPaperPlaneTriangles(
    frame: np.ndarray,
    object_vision_spec: ObjectVisionSpec,
) -> tuple[bool, Detection, Measurement]:

    detection = findSingleObjectUsingBestTriangleGroup(frame, object_vision_spec,)

    if detection is None:
        return failedDetectionResult()

    measurement = createMeasurementUsingTriangleGroup(detection, object_vision_spec,)

    return True, detection, measurement


def failedDetectionResult() -> tuple[bool, Detection, Measurement]:
    detection = Detection(None, None, None, None, [],)
    measurement = Measurement(None, None, None, None, None, None,)
    return False, detection, measurement


def drawDetection(frame: np.ndarray, detection: Detection,) -> None:

    if (detection.u is None or detection.v is None or detection.px_w is None or detection.px_h is None):
        return

    x_min = int(round(detection.u - detection.px_w / 2.0))
    y_min = int(round(detection.v - detection.px_h / 2.0))
    x_max = int(round(detection.u + detection.px_w / 2.0))
    y_max = int(round(detection.v + detection.px_h / 2.0))

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2,)

    cv2.circle(
        frame,
        (int(round(detection.u)), int(round(detection.v)),),
        radius=5, color=(0, 255, 0), thickness=-1,
    )

    # todo: change below triangles implementation as needed...
    for triangle in detection.triangles:
        vertices_px = triangle.vertices_px.astype(
            np.int32,
        )

        color_spec = COLOR_SPECS[
            triangle.color_id
        ]

        cv2.polylines(
            frame,
            [vertices_px.reshape(-1, 1, 2)],
            isClosed=True,
            color=color_spec.draw_bgr,
            thickness=2,
        )

        for vertex_u, vertex_v in vertices_px:
            cv2.circle(
                frame,
                (int(vertex_u), int(vertex_v)),
                radius=4,
                color=color_spec.draw_bgr,
                thickness=-1,
            )


def findSingleObjectUsingLargestColorBlob(frame: np.ndarray, object_vision_spec: ObjectVisionSpec) -> Detection | None:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8,)

    # Include every HSV range belonging to every configured object color.
    for color_id in object_vision_spec.color_ids:
        color_spec = COLOR_SPECS[color_id]
        for lower_hsv, upper_hsv in color_spec.hsv_ranges:
            color_mask = cv2.inRange(hsv_frame,lower_hsv, upper_hsv,)
            combined_mask = cv2.bitwise_or(combined_mask, color_mask,)

    combined_mask = cv2.medianBlur(combined_mask, 5) # apply blur

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel) # remove small random white specks
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel) # fil small black holes/gaps 

    contours, heirarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key = cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < object_vision_spec.minimum_contour_area_px:
        return None
    
    u, v, px_w, px_h = cv2.boundingRect(largest_contour)
    u = u + px_w/2.0
    v = v + px_h/2.0

    return Detection(u, v, px_w, px_h, )


def createMeasurementUsingTriangleGroup(
    detection: Detection,
    object_vision_spec: ObjectVisionSpec,
) -> Measurement:

    # Todo: replace this temporary implementation ...
    #
    #





    return Measurement(None, None, None, None, None, None,)


def findSingleObjectUsingBestTriangleGroup(frame: np.ndarray, object_vision_spec: ObjectVisionSpec,) -> Detection | None:

    # Todo: implement this...
    #
    #



    raise NotImplementedError

