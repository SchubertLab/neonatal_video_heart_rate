import os
import cv2
import pickle
import numpy as np
import pandas as pd


# Functions
def preprocess_roi_pkl_to_st_maps(experiment_data_path,
                                  st_maps_signal_type_dict,
                                  st_map_rois, mkt_name,
                                  save_path, save_file_name='_st_map_',
                                  save_file_extension='.npy',
                                  ):
    # Read files in experiment folder
    part_files = os.listdir(experiment_data_path + mkt_name)
    part_files = sorted([p for p in part_files if "mean-signals.pkl" in p])
    part_files = [p for p in part_files if "old" not in p]

    for part in part_files:
        full_path = os.path.join(experiment_data_path + mkt_name, part)
        print(part)
        with open(full_path, 'rb') as f:
            data = pickle.load(f)

        dict_output = {
            'timestamp': [],
            'time_sec_start': [],
            'time_sec_end': [],
            'st_map_name': [],
            'st_map_path': [],
        }

        temp_signal_type = data['signals_ycrcb']
        roi_list_from_data = list(data['signals_ycrcb'][0].keys())

        # for each window
        valid_window_counter = 0
        for window_i in range(len(temp_signal_type)):

            # Timestamps to sec
            # get timestamp relative to the first one, divide by 1e6 to get seconds
            start_timestamp = (data['timestamps'][window_i] - data['timestamps'][0]) / 1e6
            # create the x-axis, sample rate = 30Hz, using the starting timestamp from before
            size_window = data['signals_bgr'][0][st_map_rois[0]]
            signal_time_sec = (np.arange(size_window.shape[0]) / 30.0) + start_timestamp
            start_time_sec = signal_time_sec[0]
            end_time_sec = signal_time_sec[-1]

            # Evaluate if the window is valid
            window_valid = True
            if start_timestamp < 0:
                window_valid = False
            if data['signals_bgr'][window_i][st_map_rois[0]].shape[0] < size_window.shape[0]:
                window_valid = False

            if window_valid:
                # Check if st-map already exists
                temp_st_map_idx = str(valid_window_counter)
                st_map_file_name = mkt_name + save_file_name + temp_st_map_idx + save_file_extension
                path_file_exists = save_path + mkt_name + '/' + st_map_file_name
                file_exists = os.path.exists(path_file_exists)
                if file_exists:
                    print(st_map_file_name)
                    break

                print('window: ', window_i)
                # count total number of channels of st-map
                st_map_n_channels = np.array(
                    [len(st_maps_signal_type_dict[x]) for x in st_maps_signal_type_dict.keys()]
                ).sum()

                # Create st-map for the window
                window_st_map_i = np.zeros((size_window.shape[0], len(st_map_rois), st_map_n_channels))

                # Fill the st-map with each region and signal type channel:
                for region_idx, region_name in enumerate(st_map_rois):
                    # Check if region exists in the dataset otherwise st-map will be zeros
                    bool_rois_exists = region_name in roi_list_from_data
                    if bool_rois_exists:
                        st_map_channel_count = 0
                        for signal_type_i in list(st_maps_signal_type_dict.keys()):
                            # temp_name_base = signal_type_i + '_' + region_name
                            temp_channel_names = st_maps_signal_type_dict[signal_type_i]
                            temp_n_channels = len(temp_channel_names)

                            for channel_i, channel_name in enumerate(temp_channel_names):
                                # Get signal from data object
                                if temp_n_channels == 3:
                                    temp_array_dataset = data[signal_type_i][window_i][region_name][:, channel_i]
                                elif temp_n_channels == 1:
                                    temp_array_dataset = data[signal_type_i][window_i][region_name][:]

                                window_st_map_i[:, region_idx, st_map_channel_count] = temp_array_dataset
                                st_map_channel_count += 1
                    else:
                        print('region does not exist')

                # Normalize st-map for saving
                for roi_idx in range(len(st_map_rois)):
                    for ch in range(st_map_n_channels):
                        scaled_channel = min_max_scaling(window_st_map_i[:, roi_idx, ch])
                        window_st_map_i[:, roi_idx, ch] = scaled_channel.flatten()

                # Save current st map
                array_path = os.path.join(save_path + mkt, st_map_file_name)
                np.save(array_path, window_st_map_i)
                valid_window_counter += 1

                # Save window df info
                dict_output['timestamp'].append(data['timestamps'][window_i])
                dict_output['time_sec_start'].append(start_time_sec)
                dict_output['time_sec_end'].append(end_time_sec)
                dict_output['st_map_name'].append(st_map_file_name)
                dict_output['st_map_path'].append(array_path)

        filename = 'df_st_maps.csv'
        save_path_full = os.path.join(save_path + mkt, filename)
        df_st_maps = pd.DataFrame.from_dict(dict_output)
        df_st_maps.to_csv(save_path_full, index=False)
        print('saved')
    return


def min_max_scaling(array_in):
    a_min = np.min(array_in)
    a_max = np.max(array_in)
    diff = a_max - a_min
    array_out = array_in - a_min
    if diff != 0:
        array_out = np.divide(array_out, diff)
    return array_out


# _______________________________________________________________________________________________________
if __name__ == '__main__':

    DATASET_PATH = ''
    ROOT_SAVE_PATH = ''
    EXPERIMENT_NAME = ''
    SAVE_PATH = ROOT_SAVE_PATH + EXPERIMENT_NAME

    MEERKAT_LIST = ['1', '2', '3', '4', '5']
    VIDEO_FPS = 30

    ST_MAPS_SIGNAL_TYPE_DICT = {
        'signals_ycrcb': ['Y', 'Cr', 'Cb'],
        'signals_ir': ['ir'],
    }

    ST_MAPS_ROIS = [
        'whole-body-100',
        'upper-torso-100',
        'face-100',
    ]

    for mkt in MEERKAT_LIST:
        print(mkt)
        preprocess_roi_pkl_to_st_maps(
            experiment_data_path=DATASET_PATH,
            st_maps_signal_type_dict=ST_MAPS_SIGNAL_TYPE_DICT,
            st_map_rois=ST_MAPS_ROIS,
            mkt_name=mkt,
            save_path=SAVE_PATH,
            save_file_name='_st_map_',
            save_file_extension='.npy',
        )
