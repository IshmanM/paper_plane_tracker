import cv2
import numpy as np
import os
import threading
from src.primary.tracking import TrackStatus, SingleObjectTracker, drawTrack
from src.primary.detection import detectSingleObject, drawDetection
from src.primary.object_vision_spec import ObjectType
import src.primary.config as config
from datetime import datetime
import time
from src.primary.comm_buffer import CommBuffer, cmd_thread_main
from src.primary.platform import Platform

from src.comm.link import UdpLink
from src.comm.network_config import(
    PRIMARY_IP, 
    ENDPOINT_IP,
    UDP_PORT,
    PRIMARY_NODE_ID,
    ENDPOINT_NODE_ID,
    DEFAULT_MAX_PACKET_BYTES
)

CAMERA_INDEX = 0

if __name__ == "__main__": 
    object_type = ObjectType.TENNIS_BALL
    # object_type = ObjectType.PAPER_PLANE_TRIANGLES

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    tracker = SingleObjectTracker(
        # params...
    )

    comm_buffer = CommBuffer()
    platform = Platform(comm_buffer=comm_buffer)

    link = UdpLink(
        local_ip=PRIMARY_IP,
        remote_ip=ENDPOINT_IP,
        port=UDP_PORT,
        local_node_id=PRIMARY_NODE_ID,
        remote_node_id=ENDPOINT_NODE_ID,
        max_packet_bytes=DEFAULT_MAX_PACKET_BYTES,
        check_remote_ip=True
    )

    # link = None # FOR DEBUG ONLY

    stop_event = threading.Event()
    cmd_thread = threading.Thread(
        target=cmd_thread_main,
        args=(comm_buffer, stop_event, link),
        daemon=True # this is a background thread
    )
    cmd_thread.start()

    os.makedirs("screenshots", exist_ok=True)

    last_detection_px_w = 0
    last_detection_px_h = 0
    last_frame = None

    tracker_paused = False      # OpenCV/tracker runs by default
    platform_paused = True      # Platform OFF by default
    
    try:
        while cap.isOpened():    
            if not tracker_paused:
                frame_time = time.perf_counter()
                ret, frame = cap.read() # doesn't always give latest frame but that's a future optimization.

                if not ret:
                    print("Possible camera failure")
                    break

                # Detect the object and produce a measurement
                object_detected, detection, measurement = detectSingleObject(frame, object_type)
    
                if object_detected:
                    last_detection_px_w = detection.px_w
                    last_detection_px_h = detection.px_h
                    drawDetection(frame, detection)

                detection_label = "No detections"
                if not object_detected:
                    detection_label = "No detection"
                else:
                    detection_label = (f"Measurement: x={measurement.x:.4f}, y={measurement.y:.4f}, z={measurement.z:.4f}")


                # Track the object state & update the platform planner
                track_status = tracker.update(object_detected, measurement, frame_time)
                platform.update(tracker=tracker)
                
                track_label = "Dead track"
                if track_status == TrackStatus.CONFIRMED or track_status == TrackStatus.TENTATIVE:
                    # rectangle is drawn based on last detected px_w, px_h. might change this...
                    drawTrack(frame, tracker.track, last_detection_px_w, last_detection_px_h,)
                    track_label = ("Confirmed" if track_status == TrackStatus.CONFIRMED else "Tentative") 
                    track_label = (
                        track_label 
                        + " track: (x: " + f"{tracker.track.x:.4f}" + ", y: " + f"{tracker.track.y:.4f}"  + ", z: " + f"{tracker.track.z:.4f}" 
                        + ", dx: " + f"{tracker.track.dx:.4f}" + ", dy: " + f"{tracker.track.dy:.4f}"  + ", dz: " + f"{tracker.track.dz:.4f}" + ")"
                    )

                
                # flip (optional) as a last step before then labelling, for viewing only 
                frame = cv2.flip(frame, 1) 
                cv2.putText(frame, detection_label, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0,255,0), thickness=1)
                cv2.putText(frame, track_label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(0,0,255), thickness=1)

                last_frame = frame.copy()
            else:
                if last_frame is None: # incase of missing frame?
                    continue

                frame = last_frame.copy()


            cv2.imshow("Webcam Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            if key != 255: # FOR DEBUG ONLY
                print(f"key pressed: {key}, char: {chr(key) if key < 128 else '?'}")

            # q = quit
            if key == ord("q"):
                print("Quitting...")
                platform.halt_triggering()
                platform.turn_off()
                stop_event.set()
                break

            # p = pause BOTH OpenCV/tracker and platform
            elif key == ord("p"):
                tracker_paused = True
                platform_paused = True
                platform.turn_off()
                print("Paused tracker/OpenCV + platform OFF")

            # r = resume OpenCV/tracker only
            elif key == ord("r"):
                tracker_paused = False
                print("Tracker/OpenCV resumed")

            # l = pause platform only
            elif key == ord("l"):
                platform_paused = True
                platform.turn_off()
                print("Platform OFF")

            # o = resume platform only if OpenCV/tracker is running
            elif key == ord("o"):
                if tracker_paused:
                    print("Cannot turn platform ON while tracker/OpenCV is paused")
                else:
                    platform_paused = False
                    platform.turn_on()
                    print("Platform ON")

            # h = halt triggering
            elif key == ord("h"):
                platform.halt_triggering()
                print("Triggering HALTED")

            # f = allow triggering
            elif key == ord("f"):
                platform.allow_triggering()
                print("Triggering allowed")

            # s = screenshot
            elif key == ord("s"):
                filename = datetime.now().strftime(f"screenshot_{object_type.name.lower()}_%Y%m%d_%H%M%S.png")
                cv2.imwrite("screenshots/primary_main_screenshots/" + filename, frame)
                print(f"Saved {filename}")

    finally:
        print("Cleaning up...")
        
        stop_event.set()

        if cmd_thread.is_alive():
            cmd_thread.join(timeout=1.0)
        
        if cmd_thread.is_alive():
            print("Warning: command thread did not stop before link close")

        try:
            link.close()
        except Exception as e:
            print(f"Failed to close UDP link: {e}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")
    
