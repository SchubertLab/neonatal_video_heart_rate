"""
Extract the signals to calculate heart rate.
NOTE:
As we are unable to provide the camera data, an unimplemented class is presented
here called FolderReader to indicate how the method works.
"""

import numpy as np
import pickle
import cv2
from tqdm import tqdm
from typing import List, Tuple
import sys


class FolderReader:
    def __init__(self, folder: str):
        """
        We store files in 30s chunks, and this class is used to wrap around them so
        it appears as though it's one large file. The folder to load from is provided,
        and the files are loaded in alphabetical order.
        """
        raise NotImplementedError()

    def n_frames(self) -> int:
        """
        Returns the total number of frames in the folder.
        """
        raise NotImplementedError()

    def read_frame(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Gets the frame at a specific index. Returns the following:
        - color image [HxWx3, uint8]
        - depth image [HxW, uint16]
        - ir image [HxW, uint16]
        - params [dict, keys including 'timestamp']
        """
        raise NotImplementedError()


# Inputs:
#   mask_file: produced from script 02
#   output_file: where to store the results
#   camera_folder: where to find the camera data.
#   rotation (optional) - how to rotate the camera data, since pose estimation
#     is always done with the baby vertical.

mask_file = sys.argv[1]
output_file = sys.argv[2]
camera_folder = sys.argv[3]
rotation_key = cv2.ROTATE_90_COUNTERCLOCKWISE
if len(sys.argv) > 4:
    key_dict = {
        "counterclockwise": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "clockwise": cv2.ROTATE_90_CLOCKWISE,
        "180": cv2.ROTATE_180,
        "none": -1,
    }
    print(f"Rotation = {sys.argv[4]}")
    rotation_key = key_dict[sys.argv[4]]


# Wrapper for cv2.rotate with an option for no rotation
def rotate(img, rk):
    if rk == -1:
        return img
    return cv2.rotate(img, rk)


reader = FolderReader(camera_folder)

with open(mask_file, "rb") as f:
    data = pickle.load(f)

locations = data["locations"]
frame_numbers = data["frame_numbers"]
frames_in_file = data["frames_in_file"]
filenames = data["filenames"]
timestamps = data["timestamps"]
rois = data["rois"]
masks = data["masks"]
mask_start_xy = data["mask_start_xy"]
mask_end_xy = data["mask_stop_xy"]
mask_indices = data["indices"]
prev_indices = np.empty(0)

bgr_results = []
depth_results = []
ir_results = []
ycrcb_results = []

# We're going to store all of the images for this window. When we go to the
# next window, copy the overlap ones forward and read only the new ones to save time.
color_images = np.empty((0, 0, 0, 0), dtype=np.uint8)
ycrcb_images = np.empty_like(color_images)
depth_images = np.empty((0, 0, 0), dtype=np.uint8)
ir_images = np.empty((0, 0, 0), dtype=np.uint8)

for i in tqdm(range(len(masks))):

    my_bgr = {}
    my_depth = {}
    my_ir = {}
    my_ycrcb = {}

    indices = mask_indices[i]
    indices = indices[indices < reader.n_frames()]

    # Read frames. For the first frame, we need to load everything.
    # Otherwise, copy some of them over and only read the new ones.
    # Don't forget to rotate them all 90 degrees anticlockwise.
    if i == 0:
        prev_indices = indices
        if (
            rotation_key == cv2.ROTATE_90_CLOCKWISE
            or rotation_key == cv2.ROTATE_90_COUNTERCLOCKWISE
        ):
            color_images = np.zeros((len(indices), 1280, 720, 3), dtype=np.uint8)
            ycrcb_images = np.zeros((len(indices), 1280, 720, 3), dtype=np.uint8)
            ir_images = np.zeros((len(indices), 1280, 720), dtype=np.uint16)
            depth_images = np.zeros((len(indices), 1280, 720), dtype=np.uint16)
        else:
            color_images = np.zeros((len(indices), 720, 1280, 3), dtype=np.uint8)
            ycrcb_images = np.zeros((len(indices), 720, 1280, 3), dtype=np.uint8)
            ir_images = np.zeros((len(indices), 720, 1280), dtype=np.uint16)
            depth_images = np.zeros((len(indices), 720, 1280), dtype=np.uint16)

        for ii, ind in enumerate(indices):
            color, depth, ir, _ = reader.read_frame(ind)
            color_images[ii, ...] = rotate(color, rotation_key)
            ycrcb_images[ii, ...] = cv2.cvtColor(
                color_images[ii, ...], cv2.COLOR_BGR2YCR_CB
            )
            depth_images[ii, ...] = rotate(depth, rotation_key)
            ir_images[ii, ...] = rotate(ir, rotation_key)
    else:
        start = indices[0] - prev_indices[0]
        color_images[: prev_indices.size - start, ...] = color_images[start:, ...]
        depth_images[: prev_indices.size - start, ...] = depth_images[start:, ...]
        ir_images[: prev_indices.size - start, ...] = ir_images[start:, ...]
        ycrcb_images[: prev_indices.size - start, ...] = ycrcb_images[start:, ...]
        for ii, ind in enumerate(indices[(prev_indices.size - start) :]):
            k = ii + prev_indices.size - start
            color, depth, ir, _ = reader.read_frame(ind)
            color_images[k, ...] = rotate(color, rotation_key)
            ycrcb_images[k, ...] = cv2.cvtColor(
                color_images[k, ...], cv2.COLOR_BGR2YCR_CB
            )
            depth_images[k, ...] = rotate(depth, rotation_key)
            ir_images[k, ...] = rotate(ir, rotation_key)
        # If the window is shorter than the previous one (it will never be larger)
        # then we're at the end of the dataset. Cut down the window size.
        if indices.size != color_images.shape[0]:
            color_images = color_images[: indices.size, ...]
            depth_images = depth_images[: indices.size, ...]
            ir_images = ir_images[: indices.size, ...]
            ycrcb_images = ycrcb_images[: indices.size, ...]
        prev_indices = indices

    for roi_name in masks[i].keys():
        # The stored masks are smaller than the image size, and I've stored the
        # x and y co-ordinates where they come from in the image. (this saves a lot of data)
        # Crop the images down using the start and stop values, then multiply by the masks.
        sub_mask = masks[i][roi_name]
        start_x = mask_start_xy[i][roi_name][0]
        start_y = mask_start_xy[i][roi_name][1]
        stop_x = mask_end_xy[i][roi_name][0]
        stop_y = mask_end_xy[i][roi_name][1]

        # Generate the mask. Add the first dimension so it multiplies nicely, then add a fourth
        # dimension for multi-channel images.
        depth_ir_mask = np.expand_dims(sub_mask, 0)
        color_mask = np.expand_dims(depth_ir_mask, 3)

        # For the ir mask size, count how many non-zero depth/ir values are within the mask.
        # We don't want to count zero-value pixels when the take the mean.
        mask_size_depth = np.sum(
            (depth_images[:, start_y:stop_y, start_x:stop_x] * depth_ir_mask) != 0,
            axis=(1, 2),
        )
        mask_size_ir = np.sum(
            (ir_images[:, start_y:stop_y, start_x:stop_x] * depth_ir_mask) != 0,
            axis=(1, 2),
        )

        # Apply mask. Sum over axes 1 and 2 (i.e. over the image). For the colour images, divide by the number of pixels in the sub-mask.
        # For the depth and ir, divide by the numbers calculated before.
        masked_color = np.sum(
            color_images[:, start_y:stop_y, start_x:stop_x, :] * color_mask, axis=(1, 2)
        ) / np.sum(sub_mask)
        masked_ycrcb = np.sum(
            ycrcb_images[:, start_y:stop_y, start_x:stop_x, :] * color_mask, axis=(1, 2)
        ) / np.sum(sub_mask)
        masked_depth = (
            np.sum(
                depth_images[:, start_y:stop_y, start_x:stop_x] * depth_ir_mask,
                axis=(1, 2),
            )
            / mask_size_depth
        )
        masked_ir = (
            np.sum(
                ir_images[:, start_y:stop_y, start_x:stop_x] * depth_ir_mask,
                axis=(1, 2),
            )
            / mask_size_ir
        )

        my_bgr[roi_name] = masked_color
        my_depth[roi_name] = masked_depth
        my_ir[roi_name] = masked_ir
        my_ycrcb[roi_name] = masked_ycrcb

    bgr_results.append(my_bgr)
    depth_results.append(my_depth)
    ir_results.append(my_ir)
    ycrcb_results.append(my_ycrcb)

output_dict = {
    "locations": locations,
    "frame_numbers": frame_numbers,
    "frames_in_file": frames_in_file,
    "filenames": filenames,
    "timestamps": timestamps,
    "rois": rois,
    "indices": mask_indices,
    "signals_bgr": bgr_results,
    "signals_ycrcb": ycrcb_results,
    "signals_depth": depth_results,
    "signals_ir": ir_results,
}
with open(output_file, "wb") as f:
    pickle.dump(output_dict, f)
