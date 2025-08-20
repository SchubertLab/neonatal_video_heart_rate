"""
This script finds the corners of the ROIs
"""

import numpy as np
import pickle
from tqdm import tqdm
from typing import Dict, List


def find_rois(keypoints: np.ndarray):
    # Extract corner positions
    if keypoints.shape[1] == 3:
        keypoints = keypoints[:, :2]
    left_shoulder = keypoints[0, :]
    right_shoulder = keypoints[1, :]
    left_hip = keypoints[2, :]
    right_hip = keypoints[3, :]

    # Get torso dimensions and spine points
    shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
    hip_mid = 0.5 * (left_hip + right_hip)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    torso_height = np.linalg.norm(shoulder_mid - hip_mid)

    # Create a co-ordinate system
    y_direction = hip_mid - shoulder_mid
    if y_direction[1] < 0:
        y_direction = -y_direction
    y_direction /= np.linalg.norm(y_direction)
    x_direction = np.asarray([-y_direction[1], y_direction[0]])

    rois = {}

    # Calculate upper torso region
    upper_torso_1 = (
        shoulder_mid
        + 0.05 * torso_height * y_direction
        - 0.625 * shoulder_width * x_direction
    )
    upper_torso_2 = (
        shoulder_mid
        + 0.05 * torso_height * y_direction
        + 0.625 * shoulder_width * x_direction
    )
    upper_torso_3 = upper_torso_1 + 0.67 * torso_height * y_direction
    upper_torso_4 = upper_torso_2 + 0.67 * torso_height * y_direction

    rois["upper-torso"] = np.asarray(
        [upper_torso_1, upper_torso_2, upper_torso_3, upper_torso_4]
    )

    # Calculate face region
    face_3 = (
        shoulder_mid
        - 0.5 * shoulder_width * x_direction
        - 0.2 * torso_height * y_direction
    )
    face_4 = (
        shoulder_mid
        + 0.5 * shoulder_width * x_direction
        - 0.2 * torso_height * y_direction
    )
    face_1 = face_3 - 0.4 * torso_height * y_direction
    face_2 = face_4 - 0.4 * torso_height * y_direction
    rois["face"] = np.asarray([face_1, face_2, face_3, face_4])

    # Calculate whole body region
    full_body_1 = (
        shoulder_mid
        - 1.25 * shoulder_width * x_direction
        - 0.75 * torso_height * y_direction
    )
    full_body_2 = (
        shoulder_mid
        + 1.25 * shoulder_width * x_direction
        - 0.75 * torso_height * y_direction
    )
    full_body_3 = full_body_1 + 2.75 * torso_height * y_direction
    full_body_4 = full_body_2 + 2.75 * torso_height * y_direction

    rois["whole-body"] = np.asarray(
        [
            full_body_1,
            full_body_2,
            full_body_3,
            full_body_4,
        ]
    )

    return rois


def shrink_rois(
    rois: Dict[str, np.ndarray], sizes: List[float]
) -> Dict[str, np.ndarray]:
    """
    Produces shrunk-down rois for each region
    """
    # Co-ordinate indices
    #  0.   1.
    #
    #
    #
    #  2.   3.
    new_rois = {}
    for roi_name in rois.keys():
        full_roi = rois[roi_name]
        for size in sizes:
            size_str = str(int(size * 100))
            new_roi_name = f"{roi_name}-{size_str}"
            v1 = (1.0 - size) / 2.0
            v2 = 1.0 - v1
            new_rois[new_roi_name] = np.asarray(
                [
                    v2 * full_roi[0] + v1 * full_roi[3],
                    v2 * full_roi[1] + v1 * full_roi[2],
                    v2 * full_roi[2] + v1 * full_roi[1],
                    v2 * full_roi[3] + v1 * full_roi[0],
                ]
            )
    return new_rois


# Input arguments:
# - data file containing the results from pose estimation, with the following keys:
#     locations - pose estimation results for each frame
#     frame_numbers - the overall frame number for each processed frame
#     frames_in_file - the frame within the sub-file
#     filenames - the name of the file from which the frame came
#     timestamps - the timestamp of each frame in microsecond unix time
# - output_file: where to put the results
import sys

data_file = sys.argv[1]
output_file = sys.argv[2]

with open(data_file, "rb") as f:
    data = pickle.load(f)

locations = data["locations"]
frame_numbers = data["frame_numbers"]
frames_in_file = data["frames_in_file"]
filenames = data["filenames"]
timestamps = data["timestamps"]

n_frames = len(timestamps)
shrink_levels = [1.0, 0.9, 0.75, 0.5]

all_rois = []
for i in tqdm(range(n_frames)):
    loc = locations[i, ...]
    rois = find_rois(loc)
    rois = shrink_rois(rois, shrink_levels)
    all_rois.append(rois)

output_dict = {
    "locations": locations,
    "frame_numbers": frame_numbers,
    "frames_in_file": frames_in_file,
    "filenames": filenames,
    "timestamps": timestamps,
    "rois": all_rois,
}
with open(output_file, "wb") as f:
    pickle.dump(output_dict, f)
