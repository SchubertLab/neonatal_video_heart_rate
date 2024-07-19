"""
This script generates masks. Masks are stored as binary images, with
start (x,y) and stop (x,y). To save file space, the mask is only given
for the area between these two points, and is considered zero everywhere else.
"""

import numpy as np
import pickle
import cv2
from tqdm import tqdm
import sys

# Arguments:
#  - region file - the output from script 01
#  - output file - where to put the outputs
region_file = sys.argv[1]
output_file = sys.argv[2]
with open(region_file, "rb") as f:
    data = pickle.load(f)


locations = data["locations"]
frame_numbers = data["frame_numbers"]
frames_in_file = data["frames_in_file"]
filenames = data["filenames"]
timestamps = data["timestamps"]
rois = data["rois"]
masks = []
mask_start_xy = []
mask_stop_xy = []
mask_indices = []

image_width = 720
image_height = 1280
window_seconds = 10

window_frames = window_seconds * 30
n_frames = len(timestamps)
for i in tqdm(range(n_frames)):
    mask_frame_number = frame_numbers[i]
    indices = np.asarray(
        list(range(mask_frame_number, mask_frame_number + window_frames)), dtype=int
    )
    mask_indices.append(indices)
    my_masks = {}
    my_mask_start = {}
    my_mask_end = {}
    for key in rois[i].keys():
        image = np.zeros((image_height, image_width), dtype=np.uint8)
        roi = rois[i][key][[0, 1, 3, 2, 0], :]
        roi = roi.astype(np.int32)
        roi = np.expand_dims(roi, 0)
        cv2.fillPoly(image, roi, color=1, lineType=cv2.LINE_8)
        cols = np.argwhere(np.sum(image, axis=0) > 0)
        rows = np.argwhere(np.sum(image, axis=1) > 0)
        start_x = np.min(cols)
        start_y = np.min(rows)
        end_x = np.max(cols) + 1
        end_y = np.max(rows) + 1
        my_masks[key] = image[start_y:end_y, start_x:end_x].copy()
        my_mask_start[key] = [start_x, start_y]
        my_mask_end[key] = [end_x, end_y]
    masks.append(my_masks)
    mask_start_xy.append(my_mask_start)
    mask_stop_xy.append(my_mask_end)

output_dict = {
    "locations": locations,
    "frame_numbers": frame_numbers,
    "frames_in_file": frames_in_file,
    "filenames": filenames,
    "timestamps": timestamps,
    "rois": rois,
    "masks": masks,
    "mask_start_xy": mask_start_xy,
    "mask_stop_xy": mask_stop_xy,
    "indices": mask_indices,
}

with open(output_file, "wb") as f:
    pickle.dump(output_dict, f)
