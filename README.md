# Paper Plane Tracker

Paper Plane Tracker is a desk-scale DIY robotics sandbox for testing whether a webcam and hobby servos can follow lightweight moving objects, such as a paper plane.

The project uses hobby hardware, soft foam darts, and supervised indoor tests to explore computer vision, Kalman filtering, UDP communication, servo control, and simple timing experiments.

## What It Does

Paper Plane Tracker tries to:

* Track lightweight moving objects with a webcam
* Estimate object motion over time
* Move a small pan/tilt platform to follow the object
* Optionally trigger a soft foam dart actuator for toy timing experiments

## System Overview

```text
Webcam
  ↓
Laptop vision + tracking code
  ↓
UDP command
  ↓
Raspberry Pi endpoint
  ↓
Pan/tilt platform + optional foam actuator
```

Telemetry is sent back from the Raspberry Pi so the laptop can monitor endpoint state during testing.

## Main Components

### Laptop

The laptop handles:

* Webcam capture
* Object detection
* Position estimation
* Kalman filter tracking
* Platform command generation
* Debug output and telemetry monitoring

### Raspberry Pi Endpoint

The Raspberry Pi handles:

* Receiving UDP commands
* Applying platform safety limits
* Driving the pan/tilt servos
* Triggering the optional foam actuator
* Sending endpoint state telemetry

## Hardware

Current/expected hardware:

* Laptop
* USB webcam
* Raspberry Pi
* PCA9685 servo driver
* Hobby servos
* Small pan/tilt platform
* Low-power foam dart actuator
* External servo power supply
* Lightweight test objects, such as paper planes or foam balls

## Skills Practiced

* Computer vision
* Moving-object tracking
* Kalman filtering
* Coordinate transforms
* Embedded systems
* Servo control
* UDP networking
* Real-time robotics software design
* Basic timing and motion prediction

## Safety

This is intended as a small indoor toy robotics project.

Use only soft foam darts in a clear test area. Keep servo ranges limited, supervise the system while testing, and disconnect power before adjusting hardware.

Do not aim the mechanism at people, pets, screens, or fragile objects.

## Status

* Object detection: in progress
* Tracking: in progress
* UDP communication: in progress
* Raspberry Pi endpoint: in progress
* Pan/tilt platform control: in progress
* Foam actuator timing demo: planned / in progress

## Future Ideas

* Add ROS 2 support
* Add a simulation mode
* Improve debug visualization
* Add unit tests
* Improve the mechanical platform
* Add a simple dashboard
* Clean up the wiring and packaging

## Overall Goal

Paper Plane Tracker is a small DIY robotics project for learning how vision, tracking, networking, embedded control, and simple physical actuation fit together in one system.
