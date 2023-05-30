# Drone movements

The purpose of the "drone_movements" module is to facilitate the planning of movements for drones equipped with Forward-Looking Sonar (FLS) cameras. These movements are essential for acquiring FLS images that enable keypoint matching between subsequent image frames. The module generates a CSV file containing the desired poses that the drone needs to reach in order to execute the planned movements.

## Background

FLS cameras have a unique construction that makes it impossible to accurately determine the angle of the reflected echo along the elevation arc (Hurtós Vilarnau, 2014). While homographies are commonly used in optical imaging to approximate viewpoint changes, similar transformations can be applied to acoustic imaging (Negahdaripour, 2012). However, due to the nature of FLS cameras, such as the typically small elevation angle, objects can quickly move out of view when the camera is in motion. Therefore, we aim to restrict the camera movements to planar rotations and translations. This approach ensures the acquisition of a suitable dataset where subsequent image frames share a high percentage of the same keypoints.

## Definition

Drone movements encompass translations (surge, sway, heave) and rotations (roll, pitch, yaw). Please refer to the image below[^1] as a visual reference. However, note that our conventions assume the "z-axis" points upwards, and the "y-axis" points to the left, rather than downwards and right, respectively.

[^1]: Image source: Hurtós Vilarnau, 2014

![drone movements](/imgs/drone_movements.png)

## ROI movements

Drone movements are implemented in a region-of-interest (ROI) fashion. The drone continuously focuses on a specific region of interest while moving along arcs of different radii within the same plane. The 2D ROI movement is implemented in the [plan_roi_poses_2d](./_planner.py?plain=1#L79) function. Additionally, the concept is extended to 3D in the [plan_roi_poses_3d](./_planner.py?plain=1#L137) function. Essentially, it involves performing 2D ROI movements but on a sequence of planes.

## Output

The generated output is in the form of a CSV file. Each row in the file corresponds to a relative movement of the drone compared to its previous state. The columns represent the following drone movements:

- `surge[m]`: Surge in meters
- `sway[m]`: Sway in meters
- `heave[m]`: Heave in meters
- `roll[deg]`: Roll in degrees
- `pitch[deg]`: Pitch in degrees
- `yaw[deg]`: Yaw in degrees

:warning: **Please note that you can obtain the correct position only by following the movements in the specified order (surge, sway, heave, etc.). Failure to do so may result in an incorrect position.**

### Example

Below is an example of 3D ROI movement planning. Note the three planes for the 2D movements and that on each plane the movements are along three arcs of different radii.

![ROI drone movements example](/imgs/plan_data_example.png)
