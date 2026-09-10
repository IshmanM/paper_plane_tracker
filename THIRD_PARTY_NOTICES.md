# Third-Party Software Notices

Paper Plane Tracker uses third-party open-source software as external Python dependencies. These dependencies are not authored by this project and remain subject to their own copyright notices and license terms.

The project source code is licensed separately under the MIT License. That license does not relicense third-party software.

## Direct Dependencies

| Dependency | Used for | Upstream license |
| --- | --- | --- |
| OpenCV | Computer vision, calibration, tracking, ArUco, and pose estimation | Apache License 2.0 for OpenCV 4.5.0 and later; 3-Clause BSD for OpenCV 4.4.0 and earlier |
| NumPy | Numerical arrays, linear algebra, estimation, and geometry | BSD 3-Clause |
| Matplotlib | Plotting and visualization utilities | Matplotlib License, BSD-compatible and PSF-based |
| ReportLab | PDF generation utilities | BSD |
| Adafruit CircuitPython PCA9685 | Raspberry Pi interface to the PCA9685 PWM controller | MIT |
| Adafruit Blinka | CircuitPython compatibility layer on Raspberry Pi, including the `board` module | MIT |
| gpiozero | Raspberry Pi digital GPIO control | BSD 3-Clause |

The exact versions used by a particular installation should be recorded in the project's dependency files. Transitive dependencies installed by these packages remain subject to their own upstream licenses.

## OpenCV Calibration Acknowledgement

The checkerboard camera-calibration workflow in `src/primary/scripts/compute_camera_calibration.py` follows the standard OpenCV camera-calibration tutorial. An acknowledgement is included directly in that source file.

OpenCV documentation: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

## Upstream Projects and License Information

- OpenCV: https://opencv.org/license/
- NumPy: https://numpy.org/doc/stable/license.html
- Matplotlib: https://matplotlib.org/stable/project/license.html
- ReportLab: https://docs.reportlab.com/developerfaqs/
- Adafruit CircuitPython PCA9685: https://github.com/adafruit/Adafruit_CircuitPython_PCA9685
- Adafruit Blinka: https://github.com/adafruit/Adafruit_Blinka
- gpiozero: https://github.com/gpiozero/gpiozero

## Redistribution

No third-party source code is intentionally vendored in this repository. If third-party packages, binaries, source files, or other materials are later copied or bundled with a release, the applicable upstream copyright notices, license texts, and other required notices should be included with that distribution.

If you believe third-party material in this repository requires attribution or licensing that is not identified here, please open an issue so it can be reviewed and corrected.
