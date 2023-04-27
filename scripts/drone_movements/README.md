# Drone movements

The script [`generate.py`](./generate.py) plans FLS movements that enables keypoint matching between subsequent image frames and stores them in a CSV file. The idea is for the drone to move towards the pose (every row is one entry) and continue with the next one as soon as it has reached the desired pose.

## Background

Because of the sonar construction, it is not possible to disambiguate the angle of the reflected echo along the elevation arc (Hurtós Vilarnau, 2014). In optical imaging it is common to use homographies to approximate viewpoint changes. Similar transformations can be used in acoustic imaging (Negahdaripour, 2012). However, due to the nature of FLS (e.g., the usually small elevation angle) an object can be quickly out of view when the camera is moved. We want to limit the camera movements to planar rotations and translations, in order to obtain a proper dataset where two images from subsequent frames include a high percentage of the same keypoints.

## Definition

Drone movements are translations (surge, sway, heave) and rotations (roll, pitch, yaw). See the image below as a reference. However, note that we assume the _z-axis pointing "upwards"_ and the _y-axis pointing to the "left"_ instead of "downwards" and "right", respectively.

![drone movements](/imgs/drone_movements.png)

## Output

The generated output is in form of a CSV file. The first six columns correspond to the absolute coordinates and orientation of the sonar camera:

- `x[m]`: x position in m
- `y[m]`: y position in m
- `z[m]`: z position in m
- `x[deg]`: rotation around x-axis in degree
- `y[deg]`: rotation around y-axis in degree
- `z[deg]`: rotation around z-axis in degree

**Note that `x[deg]`, `y[deg]` and `z[deg]` are obtained from [`scipy.spatial.transform.Rotation.as_rotvec()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_rotvec.html)!**

The remaining six columns correspond to the relative movements of the drone compared to the previous state (row in the table):

- `surge[m]`: surge in m
- `sway[m]`: sway in m
- `heave[m]`: heave in m
- `roll[deg]`: roll in degree
- `pitch[deg]`: pitch in degree
- `yaw[deg]`: yaw in degree

:warning: **Note that you only obtain the correct position if you follow movements one after another (first surge, then sway, etc.). Otherwise you may end up in another position!**

### Dummy example

[`drone_move_dummy.csv`](./drone_move_dummy.csv) is an examplary CSV file whose movements can be visualized as follows:

![dummy example](./drone_move_dummy.png)
