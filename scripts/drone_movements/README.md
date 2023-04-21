# Drone movements

The script [`generate.py`](./generate.py) plans FLS movements that enables keypoint matching between subsequent image frames and stores them in a CSV file. The idea is for the drone to move towards the pose (every row is one entry) and continue with the next one as soon as it has reached the desired pose.

## Background

Because of the sonar construction, it is not possible to disambiguate the angle of the reflected echo along the elevation arc (Hurtós Vilarnau, 2014). In optical imaging it is common to use homographies to approximate viewpoint changes. Similar transformations can be used in acoustic imaging (Negahdaripour, 2012). However, due to the nature of FLS (e.g., the usually small elevation angle) an object can be quickly out of view when the camera is moved. We want to limit the camera movements to planar rotations and translations, in order to obtain a proper dataset where two images from subsequent frames include a high percentage of the same keypoints.

## Definition

Drone movements are translations (surge, sway, heave) and rotations (roll, pitch, yaw). See the image below as a reference. However, note that we assume the _z-axis pointing "upwards"_ and the _y-axis pointing to the "left"_ instead of "downwards" and "right", respectively.

![drone movements](/imgs/drone_movements.png)

## Output

The generated output is in form of a CSV file with the following columns:

- `x[mm]`: x position in mm
- `y[mm]`: y position in mm
- `z[mm]`: z position in mm
- `x[deg]`: rotation around x-axis in degree
- `x[deg]`: rotation around y-axis in degree
- `x[deg]`: rotation around z-axis in degree

The filename indicates the initial pitch and roll of the drone.

### Dummy example

[`dummy_pitch=0deg_roll=0deg.csv`](./dummy_pitch%3D0deg_roll%3D0deg.csv) is an examplary CSV file whose movements can be visualized as follows:

![dummy example](./dummy_pitch%3D0deg_roll%3D0deg.png)
