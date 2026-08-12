import time
from collections import deque

import cv2
import numpy as np

import src.primary.config as config


WINDOW_NAME = "Camera settings test"
BRIGHTNESS_HISTORY_FRAMES = 120


def applyCameraSettings(camera: cv2.VideoCapture, auto_exposure: bool, auto_wb: bool, autofocus: bool,
                        exposure: float, gain: float, wb_temperature: float, focus: float) -> None:
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exposure else 0.25)
    if not auto_exposure:
        camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
        camera.set(cv2.CAP_PROP_GAIN, gain)

    camera.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_wb else 0)
    if not auto_wb:
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)

    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    if not autofocus:
        camera.set(cv2.CAP_PROP_FOCUS, focus)


def printCameraSettings(camera: cv2.VideoCapture) -> None:
    print(f"Backend: {camera.getBackendName()}")
    print(f"Resolution: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {camera.get(cv2.CAP_PROP_FPS):.1f}")
    print(f"Auto exposure readback: {camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"Exposure: {camera.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Gain: {camera.get(cv2.CAP_PROP_GAIN)}")
    print(f"Auto WB: {camera.get(cv2.CAP_PROP_AUTO_WB)}")
    print(f"WB temperature: {camera.get(cv2.CAP_PROP_WB_TEMPERATURE)}")
    print(f"Autofocus: {camera.get(cv2.CAP_PROP_AUTOFOCUS)}")
    print(f"Focus: {camera.get(cv2.CAP_PROP_FOCUS)}")
    print(f"Brightness: {camera.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {camera.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {camera.get(cv2.CAP_PROP_SATURATION)}")


def main() -> None:
    camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    camera.set(cv2.CAP_PROP_FPS, config.FPS)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    auto_exposure = config.CAMERA_AUTO_EXPOSURE
    exposure = config.CAMERA_EXPOSURE
    gain = config.CAMERA_GAIN
    auto_wb = config.CAMERA_AUTO_WHITE_BALANCE
    wb_temperature = config.CAMERA_WHITE_BALANCE_TEMPERATURE
    autofocus = config.CAMERA_AUTOFOCUS
    focus = config.CAMERA_FOCUS

    applyCameraSettings(camera, auto_exposure, auto_wb, autofocus, exposure, gain, wb_temperature, focus)
    printCameraSettings(camera)
    print("Controls:")
    print("  Q/Esc quit | P native camera settings | R reapply config | I print readback")
    print("  A auto exposure | [/ ] exposure -/+1 | -/= gain -/+1")
    print("  W auto WB | ,/. WB temperature -/+100")
    print("  F autofocus | ;/' focus -/+5")
    print("Keep the camera still while watching the brightness range to check for flicker.")

    brightness_history = deque(maxlen=BRIGHTNESS_HISTORY_FRAMES)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera.")

            mean_brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            brightness_history.append(mean_brightness)
            brightness_range = max(brightness_history) - min(brightness_history) if brightness_history else 0.0

            lines = [
                f"AE: {'AUTO' if auto_exposure else 'MANUAL'}  exposure={exposure:.0f}  gain={gain:.0f}",
                f"WB: {'AUTO' if auto_wb else 'MANUAL'}  temperature={wb_temperature:.0f}",
                f"Focus: {'AUTO' if autofocus else 'MANUAL'}  focus={focus:.0f}",
                f"Mean brightness={mean_brightness:.1f}  rolling range={brightness_range:.1f}",
                "P=settings  A/W/F=toggle auto  I=readback  Q/Esc=quit",
            ]

            for i, line in enumerate(lines):
                y = 24 + 24*i
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("p"):
                camera.set(cv2.CAP_PROP_SETTINGS, 1)
            elif key == ord("i"):
                printCameraSettings(camera)
            elif key == ord("r"):
                auto_exposure = config.CAMERA_AUTO_EXPOSURE
                exposure, gain = config.CAMERA_EXPOSURE, config.CAMERA_GAIN
                auto_wb, wb_temperature = config.CAMERA_AUTO_WHITE_BALANCE, config.CAMERA_WHITE_BALANCE_TEMPERATURE
                autofocus, focus = config.CAMERA_AUTOFOCUS, config.CAMERA_FOCUS
                applyCameraSettings(camera, auto_exposure, auto_wb, autofocus, exposure, gain, wb_temperature, focus)
            elif key == ord("a"):
                auto_exposure = not auto_exposure
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exposure else 0.25)
                if not auto_exposure:
                    camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
                    camera.set(cv2.CAP_PROP_GAIN, gain)
            elif key == ord("["):
                exposure -= 1
                auto_exposure = False
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif key == ord("]"):
                exposure += 1
                auto_exposure = False
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif key == ord("-"):
                gain -= 1
                camera.set(cv2.CAP_PROP_GAIN, gain)
            elif key == ord("="):
                gain += 1
                camera.set(cv2.CAP_PROP_GAIN, gain)
            elif key == ord("w"):
                auto_wb = not auto_wb
                camera.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_wb else 0)
                if not auto_wb:
                    camera.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)
            elif key == ord(","):
                wb_temperature -= 100
                auto_wb = False
                camera.set(cv2.CAP_PROP_AUTO_WB, 0)
                camera.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)
            elif key == ord("."):
                wb_temperature += 100
                auto_wb = False
                camera.set(cv2.CAP_PROP_AUTO_WB, 0)
                camera.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temperature)
            elif key == ord("f"):
                autofocus = not autofocus
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
                if not autofocus:
                    camera.set(cv2.CAP_PROP_FOCUS, focus)
            elif key == ord(";"):
                focus -= 5
                autofocus = False
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                camera.set(cv2.CAP_PROP_FOCUS, focus)
            elif key == ord("'"):
                focus += 5
                autofocus = False
                camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                camera.set(cv2.CAP_PROP_FOCUS, focus)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nFinal values to copy into config.py:")
    print(f"CAMERA_AUTO_EXPOSURE = {auto_exposure}")
    print(f"CAMERA_EXPOSURE = {exposure:.1f}")
    print(f"CAMERA_GAIN = {gain:.1f}")
    print(f"CAMERA_AUTO_WHITE_BALANCE = {auto_wb}")
    print(f"CAMERA_WHITE_BALANCE_TEMPERATURE = {wb_temperature:.1f}")
    print(f"CAMERA_AUTOFOCUS = {autofocus}")
    print(f"CAMERA_FOCUS = {focus:.1f}")


if __name__ == "__main__":
    main()