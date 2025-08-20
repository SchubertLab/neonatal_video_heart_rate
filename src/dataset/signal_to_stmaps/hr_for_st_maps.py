import os
import numpy as np
import pandas as pd

# Data Config
ROOT_SAVE_PATH = ''
EXPERIMENT_NAME = ''
ST_MAPS_PATH_ROOT = ROOT_SAVE_PATH + EXPERIMENT_NAME

READ_PROCESSED_PATH_ROOT = ''
READ_LABELS_PATH_ROOT = ''

MKTS_EVAL = ['1', '2', '3', '4', '5']
PART = ''
OFFSET_MONITOR_TO_CAMERA_SEC = 0
WINDOW_INTERVENTIONS_SEC = 20


def interventions_array(df_labels, window_sample_sec):
    array_interventions = []
    intervention_start = False
    for index, row in df_labels.iterrows():
        label_row = row['Region_type']
        if label_row == 'Intervention' and not intervention_start:
            intervention_start = True
            tuple_start = row['time_signal_sec']
            tuple_end = row['time_signal_sec'] + window_sample_sec
        elif label_row != 'Intervention' and intervention_start:
            intervention_start = False
            tuple_end = row['time_signal_sec'] - window_sample_sec
            array_interventions.append([tuple_start, tuple_end])
    return array_interventions


for idx_mkt, MKT in enumerate(MKTS_EVAL):
    # Read st_maps window csv files
    ST_MAPS_FILE_NAME = 'df_st_maps.csv'
    ST_MAPS_PATH = ST_MAPS_PATH_ROOT + MKT + '/' + ST_MAPS_FILE_NAME
    df_st_map_raw = pd.read_csv(ST_MAPS_PATH)

    # time of the video with offset to monitor
    time_st_map_sec = df_st_map_raw['time_sec_end'] + OFFSET_MONITOR_TO_CAMERA_SEC
    df_st_map_raw['time_st_map_sec'] = time_st_map_sec
    df_st_map = df_st_map_raw[df_st_map_raw['time_st_map_sec'] > 0]

    # Read Monitor Signals
    VITALS_FILE_NAME = PART + '_' + MKT + '_bpm_vital_signs.csv'
    df_vital_signs_raw = pd.read_csv(READ_PROCESSED_PATH_ROOT + MKT + '/' + VITALS_FILE_NAME)
    df_vital_signs = df_vital_signs_raw[df_vital_signs_raw['signal_seconds'] > 0]

    time_monitor_sec = df_vital_signs['signal_seconds'].to_numpy()
    bpm_ecg_monitor = df_vital_signs['signal_ecg'].to_numpy()

    # Read Interventions
    INTERVENTIONS_FILE_NAME = READ_LABELS_PATH_ROOT + MKT + '/' + MKT + '_annotation_position_time.csv'
    df_intervention_labels = pd.read_csv(INTERVENTIONS_FILE_NAME, sep=';')
    temp_time_stamp = (df_intervention_labels['timestamp'] - df_intervention_labels['timestamp'][0]) / 1e6
    df_intervention_labels['time_signal_sec'] = temp_time_stamp + (np.arange(len(temp_time_stamp)) / 60.0)
    array_labels = interventions_array(
        df_labels=df_intervention_labels,
        window_sample_sec=WINDOW_INTERVENTIONS_SEC,
    )

    # Resample signal to compare - get the nearest ecg to the video values
    bpm_monitor_nearest = []
    time_monitor_nearest = []
    intervention_bool = []

    # Filter st-map so that is smaller then monitor end
    df_st_map = df_st_map[df_st_map['time_sec_end'] < time_monitor_sec[-1]]

    # iter rows of st-maps
    for index, row in df_st_map.iterrows():
        i_st_map_time = row['time_st_map_sec']
        residual = abs(time_monitor_sec - i_st_map_time).min()
        nearest_idx = np.where(abs(time_monitor_sec - i_st_map_time) == residual)[0]
        # compare to nearest bpm and time value
        if residual < 0.1:
            bpm_monitor_nearest.append(bpm_ecg_monitor[nearest_idx][0])
            time_monitor_nearest.append(time_monitor_sec[nearest_idx][0])

        # Add intervention label
        for intervention in array_labels:
            start_min = (intervention[0] + OFFSET_MONITOR_TO_CAMERA_SEC)
            end_min = (intervention[1] + OFFSET_MONITOR_TO_CAMERA_SEC)
            # if start_min < last_data_sample:
            if start_min < i_st_map_time < end_min:
                bool_interv = True
                break
            else:
                bool_interv = False

        if residual < 0.1:
            intervention_bool.append(bool_interv)

    df_result_temp_mkt = df_st_map
    df_result_temp_mkt.insert(0, 'bpm_monitor', bpm_monitor_nearest, True)
    df_result_temp_mkt.insert(1, 'time_monitor_sec', time_monitor_nearest, True)
    df_result_temp_mkt.insert(2, 'intervention', intervention_bool, True)

    # Save df
    HR_FILE_NAME = 'hr_st_maps.csv'
    SAVE_ST_MAPS_PATH = ST_MAPS_PATH_ROOT + MKT + '/' + HR_FILE_NAME
    df_result_temp_mkt.to_csv(SAVE_ST_MAPS_PATH, index=False)
    print('saved')
