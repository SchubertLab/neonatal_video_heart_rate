import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import src.dataset.signal_transforms as signal_transforms


class DataLoaderRhythmNet(Dataset):
    """
        Dataset class for RhythmNet
        input: df_dataloader with paths to st_maps and labels
        output: st_maps and hr_targets
    """

    def __init__(self, df_dataloader, n_gru_inputs=6, transform=None):
        self.maps = None
        self.n_gru_inputs = int(n_gru_inputs)
        self.target_labels_df = df_dataloader
        self.transform = transform

    def __len__(self):
        # We subtract the length of the samples for the GRU input
        return len(self.target_labels_df) - self.n_gru_inputs

    def __getitem__(self, index):
        # Load the maps for video at from 'index' to 'index'+n_gru_inputs
        maps_stack = []
        for n in range(self.n_gru_inputs):
            # name of the st_file
            stack_index = index + n
            temp_st_map_path = self.target_labels_df['st_map_path'].iloc[stack_index]
            temp_st_map = np.load(temp_st_map_path)
            if self.transform:
                temp_st_map = self.transform(temp_st_map)
            maps_stack.append(temp_st_map)

        self.maps = np.array(maps_stack)
        target_hr = self.target_labels_df["bpm_monitor"].iloc[index:index + self.n_gru_inputs].values

        return {
            "st_maps": torch.tensor(self.maps, dtype=torch.float),
            "target": torch.tensor(target_hr, dtype=torch.float)
        }


def build_dataframe_experiment(list_participants, ratio_train_val_test,
                               dataset_path, hr_filename,
                               exclude_interventions=True,
                               ):
    df_experiment = None
    for p in list_participants:
        temp_path = os.path.join(dataset_path, p, hr_filename)
        df_temp_raw = pd.read_csv(temp_path)
        # Exclude interventions
        if exclude_interventions:
            df_temp = df_temp_raw[df_temp_raw['intervention'] == False]
        else:
            df_temp = df_temp_raw

        if ratio_train_val_test[0] == 1:
            label_list = ['train'] * len(df_temp)
        else:
            train_size = int(len(df_temp) * ratio_train_val_test[0])
            val_size = int(len(df_temp) * ratio_train_val_test[1])
            test_size = len(df_temp) - (train_size + val_size)
            label_list = ['train'] * train_size + ['val'] * val_size + ['test'] * test_size
        df_temp['train_val_test'] = label_list
        df_temp['mkt'] = [p] * len(df_temp)
        df_temp['st_map_path'] = [os.path.join(dataset_path, p, x) for x in df_temp['st_map_name']]

        if df_experiment is None:
            df_temp['exp_idx'] = np.arange(0, len(df_temp))
            df_experiment = df_temp
        else:
            df_temp['exp_idx'] = np.arange(0, len(df_temp)) + len(df_experiment)
            df_experiment = pd.concat([df_experiment, df_temp])
    return df_experiment


def build_df_experiment_leave_out(
        list_train_participants,
        ratio_train_val,
        leave_out_test_participants,
        dataset_path,
        hr_filename,
        exclude_interventions=True,
):
    df_experiment = None
    for p in list_train_participants:
        temp_path = os.path.join(dataset_path, p, hr_filename)
        df_temp_raw = pd.read_csv(temp_path)
        # Exclude interventions
        if exclude_interventions:
            df_temp = df_temp_raw[df_temp_raw['intervention'] == False]
        else:
            df_temp = df_temp_raw

        train_size = int(len(df_temp) * ratio_train_val[0])
        val_size = int(len(df_temp) - train_size)
        label_list = ['train'] * train_size + ['val'] * val_size

        df_temp['train_val_test'] = label_list
        df_temp['mkt'] = [p] * len(df_temp)
        df_temp['st_map_path'] = [os.path.join(dataset_path, p, x) for x in df_temp['st_map_name']]

        if df_experiment is None:
            df_temp['exp_idx'] = np.arange(0, len(df_temp))
            df_experiment = df_temp
        else:
            df_temp['exp_idx'] = np.arange(0, len(df_temp)) + len(df_experiment)
            df_experiment = pd.concat([df_experiment, df_temp])

    # add leave-one-out participant as test
    for p_test in leave_out_test_participants:
        temp_path = os.path.join(dataset_path, p_test, hr_filename)
        df_temp_raw = pd.read_csv(temp_path)
        # Exclude interventions
        if exclude_interventions:
            df_temp = df_temp_raw[df_temp_raw['intervention'] == False]
        else:
            df_temp = df_temp_raw

        test_size = int(len(df_temp))
        label_list_test = ['test'] * test_size
        df_temp['train_val_test'] = label_list_test
        df_temp['mkt'] = [p_test] * len(df_temp)
        df_temp['st_map_path'] = [os.path.join(dataset_path, p_test, x) for x in df_temp['st_map_name']]
        df_temp['exp_idx'] = np.arange(0, len(df_temp)) + len(df_experiment)
        df_experiment = pd.concat([df_experiment, df_temp])
    return df_experiment


if __name__ == '__main__':
    TEST_DATA_LOADER = False

    if TEST_DATA_LOADER:
        # Config
        list_mkts_train_val = ['1', '2', '3', '4']
        list_mkts_test = ['5']
        root = ''
        dataset_path = root + ''
        hr_filename = 'hr_st_maps.csv'
        n_gru_inputs = 6

        # Build experiment using cross validation
        df_experiment = build_df_experiment_leave_out(
            list_train_participants=list_mkts_train_val,
            ratio_train_val=[0.7, 0.3],
            leave_out_test_participants=list_mkts_test,
            dataset_path=dataset_path,
            hr_filename=hr_filename,
            exclude_interventions=True,
        )

        df_experiment_train = df_experiment[df_experiment['train_val_test'] == 'train'].sort_values(by=['exp_idx'])
        df_experiment_val = df_experiment[df_experiment['train_val_test'] == 'val'].sort_values(by=['exp_idx'])
        df_experiment_test = df_experiment[df_experiment['train_val_test'] == 'test'].sort_values(by=['exp_idx'])

        # Create Data Transforms
        composed_transforms = transforms.Compose([
            signal_transforms.RandomGaussianNoise(),
            signal_transforms.RandomSignalFlip(),
            signal_transforms.RandomSlopeNoise(),
            signal_transforms.RandomStepNoise(),
            signal_transforms.MinMaxNorm(),
        ])

        # Data Loader Class Object
        train_set = DataLoaderRhythmNet(
            df_dataloader=df_experiment_train,
            n_gru_inputs=n_gru_inputs,
            transform=composed_transforms,
        )
        # Dataset
        train_dataloader = DataLoader(
            train_set,
            batch_size=8,
            shuffle=True
        )

        # Display image and label.
        train_objects = next(iter(train_dataloader))
        st_maps_batch = train_objects['st_maps']
        labels_batch = train_objects['target']

        for i in range(n_gru_inputs):
            n_sample = i
            print(labels_batch[2, n_sample])

            fig = plt.figure(figsize=(5, 10))
            plt.pcolormesh(st_maps_batch[2, n_sample, :, :, 0], cmap='cool')
            plt.xlabel('ROI')
            plt.xticks(
                [0.5, 1.5, 2.5],
                ['whole-body-',
                 'upper-torso',
                 'face']
            )
            plt.ylabel('Frames')
            plt.savefig('figure.pdf')
            plt.show()

            fig = plt.figure(figsize=(5, 5))
            plt.plot(st_maps_batch[2, n_sample, :, 0, 0], 'b')
