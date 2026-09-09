# Paper Plane Tracker

Paper Plane Tracker is a desk-scale DIY robotics sandbox for experimenting with real-time computer vision, 3D motion estimation, and vision-guided mechatronics using lightweight moving objects such as paper planes and tennis balls.

The project combines a webcam, laptop-based perception and tracking, a Raspberry Pi endpoint, and a hobby pan/tilt platform. Soft foam darts can also be used in supervised indoor tests as a simple physical timing demonstration.

## What It Does

Paper Plane Tracker can:

* Detect lightweight moving objects such as paper planes and tennis balls
* Estimate object position and motion over time
* Maintain a filtered 3D track using a Kalman filter
* Predict short-horizon object motion for platform-following and timing experiments
* Transform camera measurements into platform-relative coordinates
* Command a motorized pan/tilt platform in real time
* Monitor system state and timing through endpoint telemetry
* Optionally trigger a low-power soft foam actuator during supervised toy experiments

## System Overview

```text
Webcam
  ↓
Computer vision
  ↓
3D measurement + tracking
  ↓
Motion prediction
  ↓
Platform command generation
  ↓
UDP
  ↓
Raspberry Pi endpoint
  ↓
Pan/tilt platform + optional foam actuator
```

Telemetry is returned from the Raspberry Pi so the laptop can monitor platform state, timing, and endpoint behavior during testing.

## Computer Vision

The current perception pipeline includes:

* Configurable object-specific vision settings
* Color-space segmentation using HSV and LAB information
* Geometric filtering for lightweight objects
* Contour and polygon analysis
* Edge and corner refinement
* Multi-stage mask cleanup
* Paper-plane-specific geometric reasoning
* Debug visualization for inspecting intermediate processing stages

The vision pipeline has been iteratively profiled and optimized for low-latency operation on ordinary webcam input.

The project also includes tooling for generating and testing fiducial markers such as ArUco markers for controlled experiments and calibration.

## Tracking and Estimation

Object measurements are passed into a 3D tracking pipeline that includes:

* Kalman filtering
* Position and velocity estimation
* Measurement validation
* Track state management
* Short-horizon motion prediction
* Tracking uncertainty estimates

Tracking uncertainty can also be used by higher-level platform logic to avoid acting on measurements that are not sufficiently reliable while allowing tracking and platform motion to continue.

## Calibration and Coordinate Systems

The system includes calibration tools for relating camera measurements to the physical pan/tilt platform.

This includes:

* Camera intrinsic calibration
* Camera-to-platform coordinate transformation
* Platform geometry definitions
* Conversion from image-space measurements to platform-relative 3D coordinates

These tools allow the perception and physical-control portions of the system to use a consistent geometric reference frame.

## Pan/Tilt Platform

The Raspberry Pi controls a small hobby pan/tilt mechanism using servo hardware.

Platform control includes:

* Pan and tilt command generation
* Servo range limits
* Command smoothing
* Rate limiting
* Settling-time handling
* Search and follow behaviors
* Platform state telemetry

The mechanical assembly is primarily built from 3D-printed parts and hobby electronics.

## Software Architecture

The project separates higher-level perception and estimation from the physical endpoint.

### Laptop

The laptop handles:

* Webcam capture
* Object detection
* 3D measurement generation
* Kalman filter tracking
* Motion prediction
* Coordinate transformations
* Platform command generation
* Debug visualization
* Runtime and latency profiling
* Telemetry monitoring

### Raspberry Pi Endpoint

The Raspberry Pi handles:

* Receiving UDP commands
* Applying platform safety limits
* Driving the pan/tilt servos
* Operating the optional foam actuator
* Reporting endpoint state through telemetry

This split allows the computationally heavier perception pipeline to run on the laptop while the Raspberry Pi remains responsible for real-time hardware I/O.

## Hardware

Current hardware includes:

* Laptop
* USB webcam
* Raspberry Pi
* PCA9685 servo driver
* Hobby servos
* 3D-printed pan/tilt platform
* External servo power supply
* Optional low-power foam dart actuator
* Lightweight test objects such as paper planes and tennis balls

## Development and Debugging Tools

The project includes several utilities for system development and experimentation, including:

* Camera calibration tools
* Camera-to-platform calibration
* Frame-by-frame vision testing
* Intermediate vision-stage visualization
* Runtime profiling
* Live timing overlays
* Configurable object vision specifications
* Fiducial-marker generation tools
* Telemetry and platform-state logging

These tools have been useful for identifying perception errors, measurement noise, timing bottlenecks, and mechanical-control issues independently.

## Skills Practiced

* Computer vision
* Image segmentation
* Geometric feature extraction
* 3D estimation
* Kalman filtering
* Motion prediction
* Uncertainty-aware robotics
* Camera calibration
* Coordinate transforms
* Embedded systems
* Servo control
* UDP networking
* 3D-printed mechanism design
* Real-time software architecture
* Performance profiling
* Hardware/software integration

## Safety

This is a small indoor hobby robotics project intended for supervised experimentation with lightweight objects and soft foam darts.

Only soft foam darts are used during actuator tests. The system is operated in a clear test area with limited servo ranges and low-energy hobby hardware.

The mechanism should not be directed toward people, pets, screens, or fragile objects. Power should be disconnected before adjusting mechanical or electrical components.

## Status

* Paper-plane detection: functional and actively being refined
* Tennis-ball detection: functional
* Lightweight-object tracking: functional
* 3D state estimation: functional
* Camera calibration: functional
* Camera-to-platform calibration: functional
* UDP communication and telemetry: functional
* Pan/tilt control: functional
* Motion prediction: functional and being refined
* Uncertainty handling: in development
* Vision runtime optimization: ongoing
* Foam actuator timing experiments: functional and being refined
* Mechanical packaging and wiring cleanup: ongoing

## Future Ideas

* Improve robustness to lighting and motion blur
* Continue improving paper-plane geometry estimation
* Improve tracking during rapidly changing flight behavior
* Expand uncertainty-aware decision logic
* Add simulation and replay tools
* Add automated regression tests for the vision pipeline
* Improve mechanical packaging and cable management
* Explore ROS 2 integration
* Add additional lightweight-object tracking experiments

## Overall Goal

Paper Plane Tracker is a personal robotics project for exploring how perception, estimation, prediction, networking, embedded control, and physical actuation fit together in a complete real-time mechatronic system.

The paper plane provides a useful test object because its appearance, orientation, speed, and flight behavior can change rapidly, making it a more interesting perception and tracking problem than a rigid object following a predetermined path.